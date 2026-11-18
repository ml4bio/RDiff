import io
import pickle
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from . import rna_constants
from . import so3_utils


def parse_chain_feats(chain_feats, scale_factor=1.):
    """Center and scale RNA atom positions around the C1' backbone atoms."""
    bb_idx = rna_constants.BB_IDX
    chain_feats['bb_mask'] = chain_feats['atom_mask'][:, bb_idx]
    bb_pos = chain_feats['atom_positions'][:, bb_idx]
    bb_center = np.sum(bb_pos, axis=0) / np.sum(chain_feats['bb_mask'] + 1e-5)
    centered_pos = chain_feats['atom_positions'] - bb_center[None, None, :]
    scaled_pos = centered_pos / scale_factor
    chain_feats['atom_positions'] = scaled_pos * chain_feats['atom_mask'][..., None]
    chain_feats['bb_positions'] = chain_feats['atom_positions'][:, bb_idx]
    return chain_feats


def move_to_np(x):
    return x.cpu().detach().numpy()


def load_nc_sparse_type_map(path: str, length: int, nc_channel: int, symmetric: bool = True):
    """Load sparse NC interaction labels into a dense [N, N] integer map."""
    loaded = np.load(path)
    nt1 = loaded['nt1']
    nt2 = loaded['nt2']
    type_num = loaded['type_num']
    nc_type_map = np.zeros((length, length), dtype=np.int64)

    for i, j, t in zip(nt1, nt2, type_num):
        if t <= 0 or t > nc_channel:
            raise ValueError(f'Invalid NC channel index {t} in {path}')
        if i < 0 or i >= length or j < 0 or j >= length:
            raise ValueError(f'Invalid residue pair ({i}, {j}) for length {length} in {path}')
        if nc_type_map[i, j] not in (0, t):
            continue

        nc_type_map[i, j] = t
        if symmetric:
            if nc_type_map[j, i] not in (0, t):
                continue
            nc_type_map[j, i] = t

    return nc_type_map


def compose_rotvec(r1, r2):
    """Compose two rotation vectors."""
    r1_mat = Rotation.from_rotvec(r1).as_matrix()
    r2_mat = Rotation.from_rotvec(r2).as_matrix()
    composed = np.einsum('...ij,...jk->...ik', r1_mat, r2_mat)
    return Rotation.from_matrix(composed).as_rotvec()


def calc_distogram(pos, min_bin, max_bin, num_bins):
    dists_2d = torch.linalg.norm(
        pos[:, :, None, :] - pos[:, None, :, :], axis=-1)[..., None]
    lower = torch.linspace(
        min_bin,
        max_bin,
        num_bins,
        device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    return ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)


def quat_to_rotvec(quat, eps=1e-6):
    flip = (quat[..., :1] < 0).float()
    quat = (-1 * quat) * flip + (1 - flip) * quat

    angle = 2 * torch.atan2(
        torch.linalg.norm(quat[..., 1:], dim=-1),
        quat[..., 0]
    )

    angle2 = angle * angle
    small_angle_scales = 2 + angle2 / 12 + 7 * angle2 * angle2 / 2880
    large_angle_scales = angle / torch.sin(angle / 2 + eps)

    small_angles = (angle <= 1e-3).float()
    rot_vec_scale = small_angle_scales * small_angles + (1 - small_angles) * large_angle_scales
    return rot_vec_scale[..., None] * quat[..., 1:]


def quat_to_rotmat(quat, eps=1e-6):
    return so3_utils.Exp(quat_to_rotvec(quat, eps))


class CPU_Unpickler(pickle.Unpickler):
    """PyTorch pickle loading workaround for CPU-only checkpoint reads."""

    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)


def read_pkl(read_path: str, verbose=True, use_torch=False, map_location=None) -> Any:
    try:
        if use_torch:
            return torch.load(read_path, map_location=map_location)
        with open(read_path, 'rb') as handle:
            return pickle.load(handle)
    except Exception as first_error:
        try:
            with open(read_path, 'rb') as handle:
                return CPU_Unpickler(handle).load()
        except Exception as second_error:
            if verbose:
                print(f'Failed to read {read_path}. First error: {first_error}\nSecond error: {second_error}')
            raise first_error
