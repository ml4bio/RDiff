"""Score network module."""
import torch
import math
from torch import nn
from torch.nn import functional as F
from data import utils as du
from data import all_atom
from model import ipa_pytorch
import functools as fn
import pdb

from data.rna_constants import RESD_NAMES

Tensor = torch.Tensor


def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size//2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding = torch.cat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class Embedder(nn.Module):

    def __init__(self, model_conf):
        super(Embedder, self).__init__()
        self._model_conf = model_conf
        self._embed_conf = model_conf.embed

        # Time step embedding
        index_embed_size = self._embed_conf.index_embed_size
        t_embed_size = index_embed_size
        node_embed_dims = t_embed_size + 1
        edge_in = (t_embed_size + 1) * 2

        # Sequence index embedding
        node_embed_dims += index_embed_size
        edge_in += index_embed_size

        # Add contact map embedding.
        self.ct_flag = hasattr(self._embed_conf, 'ct_embed_size')
        if self.ct_flag:
            self.ct_mlp = nn.Sequential(
                nn.Linear(self._embed_conf.ct_in_ch, self._embed_conf.ct_embed_size),
                nn.SiLU(),
                nn.Linear(self._embed_conf.ct_embed_size, self._embed_conf.ct_embed_size),
            )
            edge_in += self._embed_conf.ct_embed_size

        self.use_nc_detail_cond = getattr(self._embed_conf, 'use_nc_detail_cond', False)
        if self.use_nc_detail_cond:
            self.nc_embed = nn.Embedding(
                self._embed_conf.nc_channel + 1,
                self._embed_conf.nc_embed_size,
                padding_idx=0,
            )
            edge_in += self._embed_conf.nc_embed_size

        node_embed_size = self._model_conf.node_embed_size
        self.node_embedder = nn.Sequential(
            nn.Linear(node_embed_dims, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.LayerNorm(node_embed_size),
        )

        if self._embed_conf.embed_self_conditioning:
            edge_in += self._embed_conf.num_bins
        edge_embed_size = self._model_conf.edge_embed_size
        self.edge_embedder = nn.Sequential(
            nn.Linear(edge_in, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.LayerNorm(edge_embed_size),
        )

        self.timestep_embedder = fn.partial(
            get_timestep_embedding,
            embedding_dim=self._embed_conf.index_embed_size
        )
        self.index_embedder = fn.partial(
            get_index_embedding,
            embed_size=self._embed_conf.index_embed_size
        )

        # Add fixed mask embedding
        if self._embed_conf.embed_fixed_mask:
            self.fixed_mask_embedder = nn.Embedding(
                2,
                index_embed_size,
            )
            self.fixed_mask_node_lin = nn.Linear(index_embed_size, node_embed_size)
            self.fixed_mask_edge_lin = nn.Linear(index_embed_size * 2, edge_embed_size)


    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res**2, -1])

    def forward(
            self,
            *,
            seq_idx,
            t,
            fixed_mask,
            self_conditioning_ca,
            rna_type=None,
            resd_type=None,
            ct_map=None,
            nc_type_map=None,
        ):
        """Embeds a set of inputs

        Args:
            seq_idx: [..., N] Positional sequence index for each residue.
            t: Sampled t in [0, 1].
            fixed_mask: mask of fixed (motif) residues.
            self_conditioning_ca: [..., N, 3] Ca positions of self-conditioning
                input.
            rna_type: rna types [..., N]
            resd_type: residue types [..., N]
            ct_map: contact map [..., N, N, 2]
            nc_type_map: nc type labels [..., N, N]

        Returns:
            node_embed: [B, N, D_node]
            edge_embed: [B, N, N, D_edge]
        """
        num_batch, num_res = seq_idx.shape

        # Set time step to epsilon=1e-5 for fixed residues.
        t = t[:, None] * (1 - fixed_mask) + 1e-5 * fixed_mask  # [B, num_res]
        fixed_mask = fixed_mask[..., None]
        prot_t_embed = self.timestep_embedder(t.reshape(-1)).reshape(num_batch, num_res, -1)
        # prot_t_embed = torch.tile(
        #     self.timestep_embedder(t)[:, None, :], (1, num_res, 1))
        prot_t_embed = torch.cat([prot_t_embed, fixed_mask], dim=-1)
        node_feats = [prot_t_embed]
        pair_feats = [self._cross_concat(prot_t_embed, num_batch, num_res)]

        # Positional index features.
        node_feats.append(self.index_embedder(seq_idx))
        rel_seq_offset = seq_idx[:, :, None] - seq_idx[:, None, :]
        rel_seq_offset = rel_seq_offset.reshape([num_batch, num_res**2])
        pair_feats.append(self.index_embedder(rel_seq_offset))

        # contact map MLP
        if self.ct_flag:
            ct_feats = self.ct_mlp(ct_map)
            pair_feats.append(ct_feats.reshape([num_batch, num_res**2, -1]))

        if self.use_nc_detail_cond:
            if nc_type_map is None:
                nc_type_map = torch.zeros(
                    (num_batch, num_res, num_res),
                    dtype=torch.long,
                    device=seq_idx.device,
                )
            nc_feats = self.nc_embed(nc_type_map.long())
            pair_feats.append(nc_feats.reshape([num_batch, num_res**2, -1]))

        # Self-conditioning distogram.
        if self._embed_conf.embed_self_conditioning:
            sc_dgram = du.calc_distogram(
                self_conditioning_ca,
                self._embed_conf.min_bin,
                self._embed_conf.max_bin,
                self._embed_conf.num_bins,
            )
            pair_feats.append(sc_dgram.reshape([num_batch, num_res**2, -1]))

        node_embed = self.node_embedder(torch.cat(node_feats, dim=-1).float())
        edge_embed = self.edge_embedder(torch.cat(pair_feats, dim=-1).float())

        if self._embed_conf.embed_fixed_mask:
            fixed_mask_embed = self.fixed_mask_embedder(fixed_mask.long().squeeze(-1))
            node_embed = node_embed + self.fixed_mask_node_lin(fixed_mask_embed)
            edge_embed = edge_embed + self.fixed_mask_edge_lin(
                self._cross_concat(fixed_mask_embed, num_batch, num_res)
            )

        edge_embed = edge_embed.reshape([num_batch, num_res, num_res, -1])
        return node_embed, edge_embed


class ScoreNetwork(nn.Module):

    def __init__(self, model_conf, diffuser):
        super(ScoreNetwork, self).__init__()
        self._model_conf = model_conf

        self.embedding_layer = Embedder(model_conf)
        self.diffuser = diffuser
        self.score_model = ipa_pytorch.IpaScore(model_conf, diffuser)

    def _apply_mask(self, aatype_diff, aatype_0, diff_mask):
        return diff_mask * aatype_diff + (1 - diff_mask) * aatype_0

    def forward(self, input_feats):
        """Forward computes the reverse diffusion conditionals p(X^t|X^{t+1})
        for each item in the batch

        Args:
            X: the noised samples from the noising process, of shape [Batch, N, D].
                Where the T time steps are t=1,...,T (i.e. not including the un-noised X^0)

        Returns:
            model_out: dictionary of model outputs.
        """

        # Frames as [batch, res, 7] tensors.
        bb_mask = input_feats['res_mask'].type(torch.float32)  # [B, N]
        fixed_mask = input_feats['fixed_mask'].type(torch.float32)
        edge_mask = bb_mask[..., None] * bb_mask[..., None, :]

        # Initial embeddings of positonal and relative indices.
        if 'rna_type' in input_feats:
            rna_type = input_feats['rna_type']
        else:
            rna_type = None

        ct_map = input_feats['ct'] if 'ct' in input_feats else None
        nc_type_map = input_feats['nc_type'] if 'nc_type' in input_feats else None

        if 'resd_type' in input_feats:
            resd_type = input_feats['resd_type']
        else:
            resd_type = None

        init_node_embed, init_edge_embed = self.embedding_layer(
            seq_idx=input_feats['seq_idx'],
            t=input_feats['t'],
            fixed_mask=fixed_mask,
            self_conditioning_ca=input_feats['sc_bb_t'],
            rna_type=rna_type,
            resd_type=resd_type,
            ct_map=ct_map,
            nc_type_map=nc_type_map,
        )
        edge_embed = init_edge_embed * edge_mask[..., None]
        node_embed = init_node_embed * bb_mask[..., None]

        # Run main network
        model_out = self.score_model(node_embed, edge_embed, input_feats)

        # Psi angle prediction
        # gt_psi = input_feats['torsion_angles_sin_cos'][..., 2, :]
        # psi_pred = self._apply_mask(
        #     model_out['psi'], gt_psi, 1 - fixed_mask[..., None])

        pred_out = {
            # 'psi': psi_pred,
            'rot_score': model_out['rot_score'],
            'trans_score': model_out['trans_score'],
        }
        rigids_pred = model_out['final_rigids']
        pred_out['rigids'] = rigids_pred.to_tensor_7()

        pred_bb_pos = all_atom.compute_bb_frame_atoms(rigids_pred).to(rigids_pred.device)
        pred_out['pred_bb_pos'] = pred_bb_pos

        if 'plddt' in model_out:
            pred_out['plddt'] = model_out['plddt']
        return pred_out
