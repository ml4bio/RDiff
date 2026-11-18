"""Pytorch script for SE(3) RNA diffusion inference and sampling.

This module provides the Experiment class for inference and sampling,
with all training-related code removed.
"""
import os
import torch
import tree
import numpy as np
import copy
import logging
from omegaconf import DictConfig
from omegaconf import OmegaConf

from tools import rigid_utils as ru
from data import se3_diffuser
from data import utils as du
from data import all_atom
from model import score_network


class Experiment:

    def __init__(self, conf: DictConfig):
        """Initialize experiment.

        Args:
            exp_cfg: Experiment configuration.
        """
        self._log = logging.getLogger(__name__)

        # Configs
        self._conf = conf
        self._exp_conf = conf.experiment
        self._diff_conf = conf.diffuser
        self._model_conf = conf.model
        self._data_conf = conf.data
        
        # Load checkpoint if specified (for inference)
        ckpt_model = None
        if conf.experiment.warm_start:
            ckpt_dir = conf.experiment.warm_start
            self._log.info(f'Loading checkpoint from: {ckpt_dir}')

            ckpt_name = conf.experiment.warm_start_pth
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            self._log.info(f'Loading checkpoint from {ckpt_path}')
            ckpt_pkl = du.read_pkl(ckpt_path, use_torch=True)
            ckpt_model = ckpt_pkl['model']

            if conf.experiment.use_warm_start_conf:
                OmegaConf.set_struct(conf, False)
                conf = OmegaConf.merge(conf, ckpt_pkl['conf'])
                OmegaConf.set_struct(conf, True)

        # Initialize experiment objects
        self._diffuser = se3_diffuser.SE3Diffuser(self._diff_conf)
        model_name = getattr(self._model_conf, 'name', 'score_network')
        if model_name == 'score_network':
            self._model = score_network.ScoreNetwork(self._model_conf, self.diffuser)
        else:
            raise ValueError(f'Unknown model name: {model_name}')

        if ckpt_model is not None:
            # Remove module prefix if it exists
            ckpt_model = {k[7:] if k.startswith('module.') else k: v for k, v in ckpt_model.items()}
            self._model.load_state_dict(ckpt_model, strict=True)

        num_parameters = sum(p.numel() for p in self._model.parameters())
        self._log.info(f'Number of model parameters {num_parameters}')
        model_size = num_parameters * 4 / 2 ** 20
        self._log.info('Model size: {:.1f}MB'.format(model_size))

    @property
    def diffuser(self):
        return self._diffuser

    @property
    def model(self):
        return self._model

    @property
    def conf(self):
        return self._conf

    def _self_conditioning(self, batch):
        """Self-conditioning for inference."""
        model_sc = self.model(batch)
        batch['sc_bb_t'] = model_sc['rigids'][..., 4:]
        return batch


    def _calc_trans_0(self, trans_score, trans_t, t):
        beta_t = self._diffuser._se3_diffuser._r3_diffuser.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        cond_var = 1 - torch.exp(-beta_t)
        return (trans_score * cond_var + trans_t) / torch.exp(-1/2*beta_t)

    def _set_t_feats(self, feats, t, t_placeholder):
        feats['t'] = t * t_placeholder
        rot_score_scaling, trans_score_scaling = self.diffuser.score_scaling(t)
        feats['rot_score_scaling'] = rot_score_scaling * t_placeholder
        feats['trans_score_scaling'] = trans_score_scaling * t_placeholder
        return feats

    def forward_traj(self, x_0, min_t, num_t):
        forward_steps = np.linspace(min_t, 1.0, num_t)[:-1]
        x_traj = [x_0]
        for t in forward_steps:
            x_t = self.diffuser.se3_diffuser._r3_diffuser.forward(
                x_traj[-1], t, num_t)
            x_traj.append(x_t)
        x_traj = torch.stack(x_traj, axis=0)
        return x_traj

    @torch.no_grad()
    def inference_fn(self, data_init, num_t=None, min_t=None, center=True, aux_traj=False,
                     self_condition=True, noise_scale=1.0, partial_t=-1):
        """Inference function.

        Args:
            data_init: Initial data values for sampling.
        """

        # Run reverse process.
        self.model.eval()
        sample_feats = copy.deepcopy(data_init)
        device = sample_feats['rigids_t'].device

        if sample_feats['rigids_t'].ndim == 2:
            t_placeholder = torch.ones((1,)).to(device)
        else:
            t_placeholder = torch.ones(
                (sample_feats['rigids_t'].shape[0],)).to(device)

        if num_t is None:
            num_t = self._data_conf.num_t
        if min_t is None:
            min_t = self._data_conf.min_t

        reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
        if partial_t > 0:
            reverse_steps = reverse_steps[reverse_steps < partial_t]
            reverse_steps = np.concatenate([np.array([partial_t]), reverse_steps])
        dt = 1 / num_t
        all_rigids = [du.move_to_np(copy.deepcopy(sample_feats['rigids_t']))]
        all_bb_rnas = []
        all_trans_0_pred = []
        all_bb_0_pred = []
        plddt_traj = []

        # self-cond
        if self._model_conf.embed.embed_self_conditioning and self_condition:
            sample_feats = self._set_t_feats(sample_feats, reverse_steps[0], t_placeholder)
            sample_feats = self._self_conditioning(sample_feats)

        # run reverse generative process
        for t in reverse_steps:
            if t > min_t:
                sample_feats = self._set_t_feats(sample_feats, t, t_placeholder)
                model_out = self.model(sample_feats)
                rot_score = model_out['rot_score']
                trans_score = model_out['trans_score']
                rigid_pred = model_out['rigids']
                if self._model_conf.embed.embed_self_conditioning:
                    sample_feats['sc_bb_t'] = rigid_pred[..., 4:]
                fixed_mask = sample_feats['fixed_mask'] * sample_feats['res_mask']
                diffuse_mask = (1 - sample_feats['fixed_mask']) * sample_feats['res_mask']
                rigids_t = self.diffuser.reverse(
                    rigid_t=ru.Rigid.from_tensor_7(sample_feats['rigids_t']),
                    rot_score=du.move_to_np(rot_score),
                    trans_score=du.move_to_np(trans_score),
                    diffuse_mask=du.move_to_np(diffuse_mask),
                    t=t,
                    dt=dt,
                    center=center,
                    noise_scale=noise_scale,
                )
            else:
                model_out = self.model(sample_feats)
                rigids_t = ru.Rigid.from_tensor_7(model_out['rigids'])
                rigid_pred = model_out['rigids']
                fixed_mask = sample_feats['fixed_mask'] * sample_feats['res_mask']
                diffuse_mask = (1 - sample_feats['fixed_mask']) * sample_feats['res_mask']
            
            sample_feats['rigids_t'] = rigids_t.to_tensor_7().to(device)
            if aux_traj:
                all_rigids.append(du.move_to_np(rigids_t.to_tensor_7()))

            # Calculate x0 prediction derived from score predictions.
            gt_trans_0 = sample_feats['rigids_t'][..., 4:]
            pred_trans_0 = rigid_pred[..., 4:]
            trans_pred_0 = diffuse_mask[..., None] * pred_trans_0 + fixed_mask[..., None] * gt_trans_0
            if aux_traj:
                # calculate bb atoms at time 0
                bb_atom_0 = all_atom.compute_bb_frame_atoms(ru.Rigid.from_tensor_7(rigid_pred))
                all_bb_0_pred.append(du.move_to_np(bb_atom_0))
                all_trans_0_pred.append(du.move_to_np(trans_pred_0))

            # calculate bb atoms at time t
            bb_atom_t = all_atom.compute_bb_frame_atoms(rigids_t)
            all_bb_rnas.append(du.move_to_np(bb_atom_t))
            if 'plddt' in model_out:
                plddt_traj.append(model_out['plddt'][0].cpu().numpy())

        # Flip trajectory so that it starts from t=0. This helps visualization.
        flip = lambda x: np.flip(np.stack(x), (0,))
        all_bb_rnas = flip(all_bb_rnas)
        if aux_traj:
            all_rigids = flip(all_rigids)
            all_trans_0_pred = flip(all_trans_0_pred)
            all_bb_0_pred = flip(all_bb_0_pred)

        ret = {
            'rna_traj': all_bb_rnas,
        }
        if 'plddt' in model_out:
            ret['plddt_traj'] = flip(plddt_traj)
        if aux_traj:
            ret['rigid_traj'] = all_rigids
            ret['trans_traj'] = all_trans_0_pred
            ret['rigid_0_traj'] = all_bb_0_pred
        return ret


# Note: This file is for inference and sampling only.
# Training functionality has been removed.
# Use inference_rdiff_str.py or similar scripts for inference.