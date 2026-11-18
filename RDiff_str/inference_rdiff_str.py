import os
import time
import tree
import numpy as np
import hydra
import torch
import logging
import random
from datetime import datetime
from tqdm import tqdm
import GPUtil
from typing import Optional, Dict
from omegaconf import DictConfig, OmegaConf
from Bio.PDB.PDBParser import PDBParser

import rdiff_str
from analysis import utils as au
from data import utils as du
from data import rna_constants
from tools import data_transform
from tools import rigid_utils as ru
from tools.contig_iterate import expand_contig_template

class Sampler:
    def __init__(self, conf: DictConfig, conf_overrides: Dict=None):
        """Initialize sampler.

        Args:
            conf: inference config.
            gpu_id: GPU device ID.
            conf_overrides: Dict of fields to override with new values.
        """
        self._log = logging.getLogger(__name__)

        # Remove static type checking.
        OmegaConf.set_struct(conf, False)

        # Prepare configs.
        self._conf = conf
        self._infer_conf = conf.inference
        self._diff_conf = self._infer_conf.diffusion
        if hasattr(self._infer_conf, 'samples'):
            self._sample_conf = self._infer_conf.samples
        else:
            self._sample_conf = None
        self._task_name = conf.inference.task_name

        self._rng = np.random.default_rng(self._infer_conf.seed)

        # Set-up accelerator
        if torch.cuda.is_available():
            if self._infer_conf.gpu_id is None:
                available_gpus = ''.join(
                    [str(x) for x in GPUtil.getAvailable(
                        order='memory', limit=8, maxLoad=1., maxMemory=1.)])
                self.device = f'cuda:{available_gpus[0]}'
            else:
                self.device = f'cuda:{self._infer_conf.gpu_id}'
        else:
            self.device = 'cpu'
        self._log.info(f'Using device: {self.device}')

        # Set-up directories.
        self._weights_path = self._infer_conf.weights_path
        output_dir = self._infer_conf.output_dir
        if self._infer_conf.name is None:
            dt_string = datetime.now().strftime("%dD_%mM_%YY_%Hh_%Mm_%Ss")
        else:
            dt_string = self._infer_conf.name
        self._output_dir = os.path.join(output_dir, dt_string)
        os.makedirs(self._output_dir, exist_ok=True)
        self._log.info(f'Saving results to {self._output_dir}')

        config_path = os.path.join(self._output_dir, 'inference_conf.yaml')
        with open(config_path, 'w') as f:
            OmegaConf.save(config=self._conf, f=f)
        self._log.info(f'Saving inference config to {config_path}')

        # Load models and experiment
        self._load_ckpt(conf_overrides)

    def _load_ckpt(self, conf_overrides):
        """Loads in model checkpoint."""
        self._log.info(f'Loading weights from {self._weights_path}')

        # Read checkpoint and create experiment.
        weights_pkl = du.read_pkl(
            self._weights_path, use_torch=True,
            map_location=self.device)

        # Merge base experiment config with checkpoint config.
        self._conf.model = OmegaConf.merge(
            self._conf.model, weights_pkl['conf'].model)
        if conf_overrides is not None:
            self._conf = OmegaConf.merge(self._conf, conf_overrides)

        # Prepare model
        self._conf.experiment.ckpt_dir = None
        self._conf.experiment.warm_start = None
        self.exp = rdiff_str.Experiment(
            conf=self._conf)
        self.model = self.exp.model

        # Remove module prefix if it exists.
        model_weights = weights_pkl['model']
        model_weights = {
            k[7:] if k.startswith('module.') else k: v
            for k, v in model_weights.items()
        }
        self.model.load_state_dict(model_weights)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.diffuser = self.exp.diffuser

    @staticmethod
    def scaffold_recenter(rigids_0, fixed_mask):
        # recenter rigids_0 based on fixed_mask and update rigids_t
        rigids_fixed = rigids_0 * fixed_mask[..., None]
        motif_pos = rigids_fixed[..., 4:]  # [N, 3]
        motif_center = torch.sum(motif_pos, 0) / torch.sum(fixed_mask + 1e-10)
        centered_motif_pos = (motif_pos - motif_center) * fixed_mask[..., None]
        rigids_fixed[..., 4:] = centered_motif_pos
        return rigids_fixed, motif_center

    @staticmethod
    def _seed_everything(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _write_fixed_fasta(pdb_path: str, fixed_seq: str):
        fasta_file = os.path.splitext(pdb_path)[0] + '.fasta'
        fasta_name = os.path.basename(fasta_file).replace('.fasta', '')
        with open(fasta_file, 'w') as f:
            f.write(f'>{fasta_name}_fixed\n')
            f.write(fixed_seq + '\n')

    @staticmethod
    def _pdb_rna_chain_ids(pdb_file: str):
        structure = PDBParser(PERMISSIVE=1, QUIET=1).get_structure('', pdb_file)
        rna_resnames = set(rna_constants.RESD_NAMES)
        chain_ids = []
        for chain in structure.get_chains():
            if any(res.get_resname().strip() in rna_resnames for res in chain):
                chain_ids.append(chain.id.upper())
        return chain_ids

    @staticmethod
    def _contig_chain_ids(residue_map):
        return sorted({
            residue_id.split('_', 1)[0].upper()
            for residue_id in residue_map.keys()
        })

    @staticmethod
    def _use_chain_aware_contigs(contig_chain_ids, pdb_rna_chain_ids):
        if len(contig_chain_ids) > 1 or len(pdb_rna_chain_ids) > 1:
            return True
        pdb_chain_set = set(pdb_rna_chain_ids)
        return len(contig_chain_ids) == 1 and contig_chain_ids[0] in pdb_chain_set

    @staticmethod
    def _rigids_from_atom_positions(input_pos):
        input_feats = {
            'all_atom_positions': torch.tensor(input_pos).double(),
        }
        input_feats = data_transform.make_backbone_frames(input_feats)
        return ru.Rigid.from_tensor_4x4(input_feats['gt_bb_rigids']).to_tensor_7()

    @staticmethod
    def _build_reindexed_motif(total_length, new_ids, input_seq, rigids_0, source_indices):
        diffuse_mask = np.ones(total_length)
        fixed_seq = ['-'] * total_length
        new_rigids_0 = torch.zeros((total_length, 7), dtype=rigids_0.dtype)

        for source_idx, new_id in zip(source_indices, new_ids):
            if new_id < 1 or new_id > total_length:
                raise ValueError(
                    f'Contig maps a motif residue to position {new_id}, '
                    f'but target_length is {total_length}.')
            if source_idx < 0 or source_idx >= len(input_seq):
                raise ValueError(
                    f'Contig references source residue index {source_idx + 1}, '
                    f'but the extracted motif sequence has length {len(input_seq)}.')
            diffuse_mask[new_id - 1] = 0.
            fixed_seq[new_id - 1] = input_seq[source_idx]
            new_rigids_0[new_id - 1] = rigids_0[source_idx]

        return new_rigids_0, diffuse_mask, ''.join(fixed_seq)

    def _build_chain_aware_contig_motif(self, pdb_file, total_length, residue_map):
        input_pos, _, input_seq = au.extract_specific_residue(
            PDBParser(PERMISSIVE=1, QUIET=1),
            pdb_file,
            residue_map.keys(),
        )
        rigids_0 = self._rigids_from_atom_positions(input_pos)
        return self._build_reindexed_motif(
            total_length=total_length,
            new_ids=residue_map.values(),
            input_seq=input_seq,
            rigids_0=rigids_0,
            source_indices=range(len(residue_map)),
        )

    def _build_single_chain_contig_motif(self, pdb_file, contigs, total_length):
        input_pos, _, input_seq = au.extract_3atom_from_pdb(
            PDBParser(PERMISSIVE=1, QUIET=1),
            pdb_file,
            self._infer_conf.ori_length,
        )
        self._log.info(f'Extracted sequence: {input_seq}')
        rigids_0 = self._rigids_from_atom_positions(input_pos)
        residue_map = au.generate_contig_residue_mapping(contigs)
        return self._build_reindexed_motif(
            total_length=total_length,
            new_ids=residue_map.values(),
            input_seq=input_seq,
            rigids_0=rigids_0,
            source_indices=[ori_id - 1 for ori_id in residue_map.keys()],
        )

    def _build_contig_motif(self, pdb_file, contigs, total_length):
        if contigs == '' or contigs is None:
            raise ValueError('Contigs must be specified for scaffolding.')

        chain_residue_map = au.generate_contig_residue_mapping(contigs, add_chain_name=True)
        if not chain_residue_map:
            raise ValueError(f'No motif residues found in contigs: {contigs}')

        contig_chain_ids = self._contig_chain_ids(chain_residue_map)
        pdb_rna_chain_ids = self._pdb_rna_chain_ids(pdb_file)
        use_chain_aware = self._use_chain_aware_contigs(contig_chain_ids, pdb_rna_chain_ids)
        self._log.info(
            f'PDB RNA chains: {pdb_rna_chain_ids}; contig chains: {contig_chain_ids}; '
            f'chain-aware mapping: {use_chain_aware}')

        if use_chain_aware:
            return self._build_chain_aware_contig_motif(
                pdb_file=pdb_file,
                total_length=total_length,
                residue_map=chain_residue_map,
            )
        return self._build_single_chain_contig_motif(
            pdb_file=pdb_file,
            contigs=contigs,
            total_length=total_length,
        )

    def _sample_scaffold(self, sample_dir, pdb_name, total_length, contigs, rigids_0, diffuse_mask, fixed_seq):
        diffuse_mask = torch.tensor(diffuse_mask).float()
        self._log.info(f'New fixed sequence: {fixed_seq}')

        fixed_mask = 1. - diffuse_mask
        if torch.sum(fixed_mask) <= 0:
            raise ValueError('Scaffolding requires at least one fixed motif residue.')
        rigids_fixed, motif_center = self.scaffold_recenter(rigids_0.clone(), fixed_mask)

        contig_str = contigs.replace('/', '_') if contigs is not None else 'no_contig'

        for i in range(self._infer_conf.num_samples):
            self._log.info(f'Sampling scaffold {i + 1}/{self._infer_conf.num_samples}')

            rigids_t = self.diffuser.sample_ref(n_samples=total_length, as_tensor_7=True)['rigids_t']
            rigids_t = (1 - fixed_mask[..., None]) * rigids_t + rigids_fixed

            init_feats = {
                'res_mask': torch.ones(total_length),
                'seq_idx': torch.arange(1, total_length + 1),
                'fixed_mask': fixed_mask,
                'sc_bb_t': np.zeros((total_length, 3)),
                'rigids_t': rigids_t,
            }

            init_feats = tree.map_structure(
                lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats)
            init_feats = tree.map_structure(
                lambda x: x[None].to(self.device), init_feats)
            sample_out = self.exp.inference_fn(
                init_feats,
                num_t=self._diff_conf.num_t,
                min_t=self._diff_conf.min_t,
                center=False,
                aux_traj=False,
                noise_scale=self._diff_conf.noise_scale,
            )
            sample_output = tree.map_structure(lambda x: x[:, 0], sample_out)
            pred_bb_pos = sample_output['rna_traj'][0]

            recenter_bb_pos = pred_bb_pos + motif_center.numpy()
            sample_file_n = os.path.join(sample_dir, f'{pdb_name}_scaffold_{total_length}_{contig_str}_{i}.pdb')
            au.write_backbone_atoms_to_pdb(recenter_bb_pos, sample_file_n, seq=fixed_seq.replace('-', 'A'))
            self._write_fixed_fasta(sample_file_n, fixed_seq)

    def run_sampling(self):
        if self._task_name == 'unconditional':
            self.unconditional_sampling()
        elif self._task_name == 'ss_conditional':
            self.ss_conditional_sampling()
        elif self._task_name == 'ss_nc_conditional':
            self.ss_nc_conditional_sampling()
        elif self._task_name == 'scaffolding':
            self.scaffolding()
        elif self._task_name == 'scaffolding_iter':
            self.scaffolding_iter()
        elif self._task_name == 'scaffolding_ss':
            self.scaffolding_ss_constrain()
        elif self._task_name == 'multichain_scaffolding':
            self.multichain_scaffolding()
        else:
            raise ValueError(f'Unknown task name {self._task_name}')

    def unconditional_sampling(self):
        all_sample_lengths = range(
            50, # min length
            301, # max length
            50, # step
        )
        n_sample_per_length = 10

        sample_dir = os.path.join(self._output_dir, 'sample')
        os.makedirs(sample_dir, exist_ok=True)

        # iter over lengths
        for sample_length in all_sample_lengths:
            self._log.info(f'Sampling length {sample_length}')
            # iter over samples with the same length
            for sample_i in range(n_sample_per_length):
                sample_file_n = os.path.join(sample_dir, f'length_{sample_length}_sample_{sample_i}')

                sample_output = self.sample(sample_length)
                pdb_path = au.write_backbone_atoms_to_pdb(sample_output['rna_traj'][0], sample_file_n)
                self._log.info(f'Done sample {sample_i}: {pdb_path}')


    def sample(self, sample_length: int, rna_type: Optional[str]=None, init_feats: Optional[Dict]=None,
               partial_t: Optional[float]=-1., ss_map: Optional=None,
               nc_type_map: Optional=None) -> Dict[str, torch.Tensor]:
        """Sample based on length.

        Args:
            sample_length: length to sample
            partial_t: partial start time to sample
            rna_type: RNA type to sample
            init_feats: initial features for sampling
            ss_map: secondary structure map for sampling [N, N, 1or2or3]
            nc_type_map: noncanonical interaction type map [N, N]

        Returns:
            Sample outputs. See rdiff_str.Experiment.inference_fn.
        """

        if init_feats is None:
            # Process motif features.
            res_mask = np.ones(sample_length)
            fixed_mask = np.zeros_like(res_mask)

            # Initialize data
            ref_sample = self.diffuser.sample_ref(
                n_samples=sample_length,
                as_tensor_7=True,
            )
            res_idx = torch.arange(1, sample_length+1)
            init_feats = {
                'res_mask': res_mask,
                'seq_idx': res_idx,
                'fixed_mask': fixed_mask,
                # torsion_angles_sin_cos': np.zeros((sample_length, 7, 2)),
                'sc_bb_t': np.zeros((sample_length, 3)),
                **ref_sample,
            }
            if ss_map is not None:
                init_feats['ct'] = ss_map
            if nc_type_map is not None:
                init_feats['nc_type'] = nc_type_map

            if rna_type is not None:
                init_feats['rna_type'] = torch.tensor(rna_constants.RNA_TYPES[rna_type]).repeat(sample_length)

        # Add batch dimension and move to GPU.
        init_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats)
        init_feats = tree.map_structure(
            lambda x: x[None].to(self.device), init_feats)

        # Run inference
        sample_out = self.exp.inference_fn(
            init_feats,
            num_t=self._diff_conf.num_t,
            min_t=self._diff_conf.min_t,
            aux_traj=False,
            noise_scale=self._diff_conf.noise_scale,
            partial_t=partial_t,
        )

        # remove batch dimension
        return tree.map_structure(lambda x: x[:, 0], sample_out)


    def ss_conditional_sampling(self):
        # process secondary structure information
        target_length = self._infer_conf.target_length
        n_samples = self._infer_conf.num_samples

        ss_db_constrain = self._infer_conf.ss_db_constrain
        assert len(ss_db_constrain) == target_length, \
            f'Length of secondary structure {len(ss_db_constrain)} does not match target length {target_length}.'
        ssmap = au.parse_cwwss_db(ss_db_constrain)
        assert ssmap.shape[0] == target_length, \
            f'Secondary structure map length {ssmap.shape[0]} does not match target length {target_length}.'

        sample_dir = self._output_dir
        os.makedirs(sample_dir, exist_ok=True)
        for i in range(n_samples):
            self._log.info(f'Sampling ss-conditional sample {i + 1}/{n_samples}')

            sample_output = self.sample(
                sample_length=target_length,
                ss_map=ssmap,
            )
            pred_bb_pos = sample_output['rna_traj'][0]  # [N_resd, 3, 3]

            # obtain plddt confidence scores
            if 'plddt_traj' in sample_output:
                plddt_mean_t = int(0.5 * sample_output['plddt_traj'].shape[0])
                confidence = np.mean(sample_output['plddt_traj'][:plddt_mean_t], axis=0)  # [N_resd,]
            else:
                confidence = None

            sample_file_n = os.path.join(sample_dir, f'ss_conditional_sample_{i}.pdb')
            au.write_backbone_atoms_to_pdb(pred_bb_pos, sample_file_n, seq=None, confidence=confidence)

    def _select_conf(self, path, default=None):
        value = OmegaConf.select(self._conf, path)
        return default if value is None else value

    @staticmethod
    def _has_conf_value(value):
        return value is not None and value != '' and str(value).lower() != 'null'

    def _load_ssmap3(self, ssmap_path):
        ssmap = np.load(ssmap_path)
        ss_channel = int(self._select_conf('data.ss_channel', self._select_conf('model.embed.ct_in_ch', 3)))

        if ssmap.ndim != 3:
            raise ValueError(f'ssmap must be 3D, got shape {ssmap.shape} from {ssmap_path}')

        if ssmap.shape[0] == ssmap.shape[1]:
            # Already [N, N, C].
            if ssmap.shape[2] < ss_channel:
                raise ValueError(
                    f'ssmap channel count {ssmap.shape[2]} is smaller than required {ss_channel}: {ssmap_path}')
            ssmap = ssmap[:, :, :ss_channel]
        elif ssmap.shape[1] == ssmap.shape[2]:
            # Training data stores ssmap3 as [C, N, N].
            if ssmap.shape[0] < ss_channel:
                raise ValueError(
                    f'ssmap channel count {ssmap.shape[0]} is smaller than required {ss_channel}: {ssmap_path}')
            ssmap = ssmap[:ss_channel].transpose(1, 2, 0)
        else:
            raise ValueError(f'Cannot infer ssmap layout from shape {ssmap.shape}: {ssmap_path}')

        if ssmap.shape[0] != ssmap.shape[1]:
            raise ValueError(f'ssmap residue dimensions must match, got {ssmap.shape}: {ssmap_path}')
        return ssmap.astype(np.float32)

    def _load_nc_type_map(self, nc_path, length):
        nc_channel = int(self._select_conf('data.nc_channel', self._select_conf('model.embed.nc_channel', 11)))
        return du.load_nc_sparse_type_map(
            nc_path,
            length=length,
            nc_channel=nc_channel,
            symmetric=True,
        )

    @staticmethod
    def _strip_target_name(target):
        return os.path.splitext(os.path.basename(target.strip()))[0]

    def _get_ss_nc_targets(self, ssmap_dir):
        target_list_path = self._select_conf('inference.target_list', None)
        if self._has_conf_value(target_list_path):
            with open(target_list_path, 'r') as f:
                target_names = [self._strip_target_name(line) for line in f if line.strip()]
        else:
            target_names = [
                os.path.splitext(f)[0]
                for f in os.listdir(ssmap_dir)
                if f.endswith('.npy')
            ]
        target_names = sorted(set(target_names))

        max_targets = self._select_conf('inference.max_targets', None)
        if max_targets is not None and int(max_targets) > 0:
            target_names = target_names[:int(max_targets)]
        return target_names

    def ss_nc_conditional_sampling(self):
        ssmap_path = self._select_conf('inference.ssmap_path', None)
        nc_sparse_path = self._select_conf('inference.nc_sparse_path', None)
        use_single_pair = self._has_conf_value(ssmap_path) or self._has_conf_value(nc_sparse_path)

        if use_single_pair:
            if not self._has_conf_value(ssmap_path) or not self._has_conf_value(nc_sparse_path):
                raise ValueError('Both inference.ssmap_path and inference.nc_sparse_path must be set for single-file inference.')
            if not os.path.exists(ssmap_path):
                raise ValueError(f'ssmap_path does not exist: {ssmap_path}')
            if not os.path.exists(nc_sparse_path):
                raise ValueError(f'nc_sparse_path does not exist: {nc_sparse_path}')
            target_specs = [(self._strip_target_name(ssmap_path), ssmap_path, nc_sparse_path)]
        else:
            condition_dir = self._infer_conf.condition_dir
            ssmap_dir = self._select_conf('inference.ssmap_dir', None)
            nc_sparse_dir = self._select_conf('inference.nc_sparse_dir', None)
            if not self._has_conf_value(ssmap_dir):
                ssmap_dir = os.path.join(condition_dir, 'ssmap3')
            if not self._has_conf_value(nc_sparse_dir):
                nc_sparse_dir = os.path.join(condition_dir, 'NC_sparse')

            if not os.path.isdir(ssmap_dir):
                raise ValueError(f'ssmap_dir does not exist: {ssmap_dir}')
            if not os.path.isdir(nc_sparse_dir):
                raise ValueError(f'nc_sparse_dir does not exist: {nc_sparse_dir}')

            target_names = self._get_ss_nc_targets(ssmap_dir)
            if len(target_names) == 0:
                raise ValueError(f'No .npy ssmap files found in {ssmap_dir}')

            target_specs = [
                (
                    target_name,
                    os.path.join(ssmap_dir, target_name + '.npy'),
                    os.path.join(nc_sparse_dir, target_name + '.npz'),
                )
                for target_name in target_names
            ]

        n_samples = self._infer_conf.num_samples
        configured_length = int(self._select_conf('inference.target_length', -1))
        self._log.info(f'Running ss+NC conditional inference for {len(target_specs)} targets')

        for target_i, (target_name, ssmap_path, nc_path) in enumerate(target_specs):
            if not os.path.exists(ssmap_path):
                raise ValueError(f'Missing ssmap file for {target_name}: {ssmap_path}')
            if not os.path.exists(nc_path):
                raise ValueError(f'Missing NC sparse file for {target_name}: {nc_path}')

            ssmap = self._load_ssmap3(ssmap_path)
            target_length = ssmap.shape[0]
            if configured_length > 0 and configured_length != target_length:
                raise ValueError(
                    f'inference.target_length={configured_length} does not match {target_name} length {target_length}')
            nc_type_map = self._load_nc_type_map(nc_path, target_length)

            sample_dir = os.path.join(self._output_dir, target_name)
            os.makedirs(sample_dir, exist_ok=True)
            self._log.info(
                f'Sampling target {target_i + 1}/{len(target_specs)}: {target_name}, length={target_length}')

            for sample_i in range(n_samples):
                self._log.info(f'Sampling {target_name} backbone {sample_i + 1}/{n_samples}')
                sample_output = self.sample(
                    sample_length=target_length,
                    ss_map=ssmap,
                    nc_type_map=nc_type_map,
                )
                pred_bb_pos = sample_output['rna_traj'][0]

                if 'plddt_traj' in sample_output:
                    plddt_mean_t = int(0.5 * sample_output['plddt_traj'].shape[0])
                    confidence = np.mean(sample_output['plddt_traj'][:plddt_mean_t], axis=0)
                else:
                    confidence = None

                sample_file_n = os.path.join(sample_dir, f'{target_name}_sample_{sample_i}.pdb')
                au.write_backbone_atoms_to_pdb(pred_bb_pos, sample_file_n, seq=None, confidence=confidence)


    def multichain_scaffolding(self):
        self.scaffolding()


    def scaffolding_iter(self):
        iter_conf = self._infer_conf.get('scaffold_iter', {}) or {}
        contig_template = iter_conf.get('contigs', None) or self._infer_conf.contigs
        length_min = int(iter_conf.get('length_min', self._infer_conf.target_length))
        length_max = int(iter_conf.get('length_max', length_min))
        length_step = int(iter_conf.get('length_step', 1))
        max_contigs_per_length = iter_conf.get('max_contigs_per_length', None)

        if length_step <= 0:
            raise ValueError('inference.scaffold_iter.length_step must be positive.')
        if length_max < length_min:
            raise ValueError('inference.scaffold_iter.length_max must be >= length_min.')
        if max_contigs_per_length is not None:
            max_contigs_per_length = int(max_contigs_per_length)
            if max_contigs_per_length <= 0:
                max_contigs_per_length = None

        length_contigs = []
        for target_length in range(length_min, length_max + 1, length_step):
            contigs = expand_contig_template(
                contig_template,
                target_length=target_length,
            )
            if max_contigs_per_length is not None and len(contigs) > max_contigs_per_length:
                sampled_indices = self._rng.choice(
                    len(contigs),
                    size=max_contigs_per_length,
                    replace=False,
                )
                contigs = [contigs[idx] for idx in sampled_indices]

            for contig in contigs:
                length_contigs.append((target_length, contig))

        if not length_contigs:
            raise ValueError(
                f'No contigs generated from template {contig_template!r} '
                f'for lengths {length_min}-{length_max}.')
        self._log.info(f'Generated {len(length_contigs)} contigs from template {contig_template!r}.')

        for target_length, contig in tqdm(length_contigs):
            seed = int(self._rng.integers(0, 1000001))
            self._log.info(f'Length: {target_length}, Contig: {contig}, Seed: {seed}')

            self._infer_conf.target_length = target_length
            self._infer_conf.contigs = contig
            self._infer_conf.seed = seed
            self._seed_everything(seed)

            self.scaffolding()


    def scaffolding(self):
        sample_dir = self._output_dir
        os.makedirs(sample_dir, exist_ok=True)

        pdb_file = self._infer_conf.target_pdb_path
        pdb_name = os.path.basename(pdb_file).replace('.pdb', '')
        total_length = self._infer_conf.target_length

        if self._infer_conf.use_contigs:
            contigs = self._infer_conf.contigs
            rigids_0, diffuse_mask, fixed_seq = self._build_contig_motif(
                pdb_file=pdb_file,
                contigs=contigs,
                total_length=total_length,
            )
        else:
            contigs = None
            input_pos, missing_mask, input_seq = au.extract_3atom_from_pdb(
                PDBParser(PERMISSIVE=1, QUIET=1),
                pdb_file,
                self._infer_conf.ori_length,
            )
            self._log.info(f'Extracted sequence: {input_seq}')
            rigids_0 = self._rigids_from_atom_positions(input_pos)
            diffuse_mask = missing_mask
            fixed_seq = input_seq

        self._sample_scaffold(
            sample_dir=sample_dir,
            pdb_name=pdb_name,
            total_length=total_length,
            contigs=contigs,
            rigids_0=rigids_0,
            diffuse_mask=diffuse_mask,
            fixed_seq=fixed_seq,
        )


    def scaffolding_ss_constrain(self):
        # extra a PDB structure, generate the missing residues in the target length while keep the secondary structure
        # extra a PDB structure, and generate the missing residues in the target length

        sample_dir = self._output_dir
        os.makedirs(sample_dir, exist_ok=True)

        # read and parse PDB files
        pdb_file = self._infer_conf.target_pdb_path
        pdb_name = os.path.basename(pdb_file).replace('.pdb', '')
        total_length = self._infer_conf.target_length

        input_pos, missing_mask, input_seq = au.extract_3atom_from_pdb(PDBParser(PERMISSIVE=1, QUIET=1),
                                                                       pdb_file, self._infer_conf.ori_length)

        self._log.info(f'Extracted sequence: {input_seq}')

        input_feats = {
            'all_atom_positions': torch.tensor(input_pos).double(),
        }
        input_feats = data_transform.make_backbone_frames(input_feats)  # gt_bb_rigids
        rigids_0 = ru.Rigid.from_tensor_4x4(input_feats['gt_bb_rigids']).to_tensor_7()

        if self._infer_conf.use_contigs:
            # rearrange the residue number (rigid order) according to the contigs
            contigs = self._infer_conf.contigs
            residue_map = au.generate_contig_residue_mapping(contigs)
            diffuse_mask = np.ones(total_length)
            fixed_seq = ['-'] * total_length
            new_rigids_0 = torch.zeros((total_length, 7), dtype=rigids_0.dtype)
            for ori_id, new_id in residue_map.items():
                diffuse_mask[new_id - 1] = 0.
                fixed_seq[new_id - 1] = input_seq[ori_id - 1]
                new_rigids_0[new_id - 1] = rigids_0[ori_id - 1]
            fixed_seq = ''.join(fixed_seq)
            rigids_0 = new_rigids_0
        else:
            diffuse_mask = missing_mask
            fixed_seq = input_seq

        diffuse_mask = torch.tensor(diffuse_mask).float()
        self._log.info(f'New fixed sequence: {fixed_seq}')

        # recenter conditional rigids based on the fixed mask
        fixed_mask = 1. - diffuse_mask
        rigids_fixed, motif_center = self.scaffold_recenter(rigids_0.clone(), fixed_mask)

        # process secondary structure information
        ss_db_constrain = self._infer_conf.ss_db_constrain
        assert len(ss_db_constrain) == total_length, \
            f'Length of secondary structure {len(ss_db_constrain)} does not match target length {total_length}.'
        ssmap = au.parse_cwwss_db(ss_db_constrain)

        # iterate over number of samples
        for i in range(self._infer_conf.num_samples):
            self._log.info(f'Sampling scaffold {i + 1}/{self._infer_conf.num_samples}')

            rigids_t = self.diffuser.sample_ref(n_samples=total_length, as_tensor_7=True)['rigids_t']
            rigids_t = (1 - fixed_mask[..., None]) * rigids_t + rigids_fixed

            # run diffusion inference
            init_feats = {
                'res_mask': torch.ones(total_length),
                'seq_idx': torch.arange(1, total_length + 1),
                'fixed_mask': fixed_mask,
                'sc_bb_t': np.zeros((total_length, 3)),
                'rigids_t': rigids_t,
                'ct': ssmap,
            }

            init_feats = tree.map_structure(
                lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats)
            init_feats = tree.map_structure(
                lambda x: x[None].to(self.device), init_feats)
            sample_out = self.exp.inference_fn(init_feats, num_t=self._diff_conf.num_t, min_t=self._diff_conf.min_t,
                                               center=False, aux_traj=False, noise_scale=self._diff_conf.noise_scale)
            sample_output = tree.map_structure(lambda x: x[:, 0], sample_out)
            pred_bb_pos = sample_output['rna_traj'][0]  # [N_resd, 3, 3]

            # remove back to original center; then add three-atoms for missing residues, save imputation results
            recenter_bb_pos = pred_bb_pos + motif_center.numpy()
            sample_file_n = os.path.join(sample_dir, f'{pdb_name}_scaffold_{total_length}_{i}.pdb')
            au.write_backbone_atoms_to_pdb(recenter_bb_pos, sample_file_n, seq=fixed_seq.replace('-', 'A'))
            self._write_fixed_fasta(sample_file_n, fixed_seq)


@hydra.main(version_base=None, config_path='config', config_name='rna_unconditional_training')
def run(conf: DictConfig) -> None:

    # Read model checkpoint.
    print('Starting inference')
    start_time = time.time()
    sampler = Sampler(conf)
    sampler.run_sampling()
    elapsed_time = time.time() - start_time
    print(f'Finished in {elapsed_time:.2f}s')

if __name__ == '__main__':
    run()
