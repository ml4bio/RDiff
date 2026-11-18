import torch
from .rigid_utils import Rotation, Rigid
import pdb

def make_backbone_frames(rna, eps=1e-8):
    """Compute backbone frames from RNA coordinates."""
    all_atom_positions = rna['all_atom_positions']

    # Compute backbone frames
    # Note: The idealized pos is
    #         ["C4'", 0, (-1.276165, 1.959880, -0.000000)], is xy-plane
    #         ["C1'", 0, (0.000000, 0.000000, 0.000000)],
    #         ["N9", 0, (1.475205, -0.000001, -0.000000)]. is pos_x_axis

    gt_bb_frames = Rigid.from_3_points(
        p_neg_x_axis=all_atom_positions[..., 2, :], # N1/N9
        origin=all_atom_positions[..., 1, :],  # C1'
        p_xy_plane=all_atom_positions[..., 0, :], # C4'
        eps=eps
    ) # [num_res]

    # Adapt backbone frame to old convention (mirror x-axis and z-axis)
    rots = torch.eye(3, dtype=all_atom_positions.dtype, device=all_atom_positions.device)
    rots = torch.tile(rots, (1, 1, 1))
    rots[..., 0, 0] = -1
    rots[..., 2, 2] = -1
    rots = Rotation(rot_mats=rots)

    gt_bb_frames = gt_bb_frames.compose(Rigid(rots, None))

    rna['gt_bb_rigids'] = gt_bb_frames.to_tensor_4x4()
    return rna

