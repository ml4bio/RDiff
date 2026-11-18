"""Utilities for calculating all atom representations."""
import torch
import pdb

from . import rna_constants
from tools import rigid_utils as ru


Rigid = ru.Rigid
Rotation = ru.Rotation
IDEALIZED_BB_POS_3 = torch.tensor(rna_constants.IDEALIZED_POS_C)


def compute_bb_frame_atoms(bb_rigids):
    """Return (C4, C1', N9/N1) atoms in the backbone frame. Shape [*, N, 3, 3]"""
    bb_null_pos = IDEALIZED_BB_POS_3[None, None, ...].to(bb_rigids.device)
    # bb_null_pos = bb_null_pos.repeat((1, bb_rigids.shape[1], 1, 1))
    pred_bb_pos = bb_rigids[..., None].apply(bb_null_pos)
    return pred_bb_pos

