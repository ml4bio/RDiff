"""SE(3) diffusion methods."""
import numpy as np
from data import so3_diffuser
from data import r3_diffuser
from scipy.spatial.transform import Rotation
# from openfold.utils import rigid_utils as ru
from tools import rigid_utils as ru
from data import utils as du
import torch
import logging
import pdb


def kabsch_align_np(coords_a, coords_b, mask):
    """
    Aligns coords_b to coords_a using Kabsch algorithm (NumPy version).

    Args:
        coords_a: Reference coordinates (shape [N, 3]).
        coords_b: Coordinates to align (shape [N, 3]).
        mask: Mask indicating valid points (shape [N, 1] or [N,]).
              Points with mask=0 are ignored in mean calculation.

    Returns:
        coords_b_aligned: Aligned coordinates of coords_b (shape [N, 3]).
    """
    # Ensure mask is shape [N, 1] for broadcasting
    if mask.ndim == 1:
        mask = mask[:, np.newaxis] # Reshape (N,) to (N, 1)

    # --- 1. Calculate centroids (weighted mean) ---
    # Calculate sum of mask for normalization
    # Add a small epsilon to prevent division by zero
    mask_sum = np.sum(mask, axis=0, keepdims=True)
    epsilon = 1e-10
    if np.all(mask_sum < epsilon):
         # If no valid points, can return original coordinates or raise an error
         print("Warning: No valid points in mask. Returning original coords_b.")
         # Or raise ValueError("No valid points provided in the mask.")
         return coords_b

    # Calculate weighted mean centroids
    coords_a_mean = np.sum(coords_a * mask, axis=0, keepdims=True) / (mask_sum + epsilon) # Shape [1, 3]
    coords_b_mean = np.sum(coords_b * mask, axis=0, keepdims=True) / (mask_sum + epsilon) # Shape [1, 3]

    # --- 2. Center coordinates ---
    coords_a_cen = coords_a - coords_a_mean # Broadcasting applies mean to all rows
    coords_b_cen = coords_b - coords_b_mean

    # --- 3. Calculate covariance matrix ---
    # Note: The original PyTorch code implies the mask affects mean calculation,
    # but not directly the covariance calculation here.
    # If the mask should also apply to covariance calculation, the code would be:
    # A = (coords_a_cen * mask).T @ (coords_b_cen * mask)
    # But to accurately translate the PyTorch structure, we use:
    A = coords_a_cen.T @ coords_b_cen # [3, N] @ [N, 3] -> [3, 3]

    # --- 4. Perform SVD ---
    # np.linalg.svd returns U, S, Vh (where Vh = Vt)
    U, S, Vt = np.linalg.svd(A)

    # --- 5. Handle possible reflections ---
    # Calculate sign of the determinant of A
    det_A = np.linalg.det(A)
    # Create correction matrix
    corr_mat = np.diag([1, 1, np.sign(det_A)])

    # --- 6. Calculate optimal rotation matrix R ---
    rotation = (U @ corr_mat) @ Vt

    # --- 7. Calculate optimal translation vector t ---
    # t = coords_a_mean - R @ coords_b_mean
    # Need to be careful with matrix shapes and transposes
    # coords_a_mean is [1, 3]
    # coords_b_mean is [1, 3], after transpose coords_b_mean.T is [3, 1]
    # rotation is [3, 3]
    # R @ coords_b_mean.T is [3, 3] @ [3, 1] -> [3, 1]
    # (R @ coords_b_mean.T).T is [1, 3]
    translation = coords_a_mean - (rotation @ coords_b_mean.T).T # Shape [1, 3]

    # --- 8. Apply rotation and translation to coords_b ---
    # coords_b_aligned = R @ coords_b^T + t^T (if t is a column vector)
    # Or coords_b_aligned = (R @ coords_b.T).T + t (if t is a row vector)
    coords_b_aligned = (rotation @ coords_b.T).T + translation # ([3,3] @ [3,N]).T + [1,3] -> [N,3] + [1,3] -> [N,3]

    # --- Another equivalent way to align: ---
    # Aligned points = R @ (original points - centroid B) + centroid A
    # coords_b_aligned = (rotation @ coords_b_cen.T).T + coords_a_mean
    # This is mathematically equivalent

    return coords_b_aligned


def _extract_trans_rots(rigid: ru.Rigid):
    rot = rigid.get_rots().get_rot_mats().cpu().numpy()
    rot_shape = rot.shape
    num_rots = np.cumprod(rot_shape[:-2])[-1]
    rot = rot.reshape((num_rots, 3, 3))
    rot = Rotation.from_matrix(rot).as_rotvec().reshape(rot_shape[:-2] +(3,))
    tran = rigid.get_trans().cpu().numpy()
    return tran, rot

def _assemble_rigid(rotvec, trans):
    rotvec_shape = rotvec.shape
    num_rotvecs = np.cumprod(rotvec_shape[:-1])[-1]
    rotvec = rotvec.reshape((num_rotvecs, 3))
    rotmat = Rotation.from_rotvec(rotvec).as_matrix().reshape(
        rotvec_shape[:-1] + (3, 3))
    return ru.Rigid(
            rots=ru.Rotation(
                rot_mats=torch.Tensor(rotmat)),
            trans=torch.tensor(trans))

class SE3Diffuser:

    def __init__(self, se3_conf):
        self._log = logging.getLogger(__name__)
        self._se3_conf = se3_conf

        self._diffuse_rot = se3_conf.diffuse_rot
        self._so3_diffuser = so3_diffuser.SO3Diffuser(self._se3_conf.so3)

        self._diffuse_trans = se3_conf.diffuse_trans
        self._r3_diffuser = r3_diffuser.R3Diffuser(self._se3_conf.r3)

    def forward_marginal(
            self,
            rigids_0: ru.Rigid,
            t: float,
            diffuse_mask: np.ndarray = None,
            as_tensor_7: bool=True,
            align_trans: bool=False,
        ):
        """
        Args:
            rigids_0: [..., N] openfold Rigid objects
            t: continuous time in [0, 1].
            align_trans: true to align translation to original coords

        Returns:
            rigids_t: [..., N] noised rigid. [..., N, 7] if as_tensor_7 is true. 
            trans_score: [..., N, 3] translation score
            rot_score: [..., N, 3] rotation score
            trans_score_norm: [...] translation score norm
            rot_score_norm: [...] rotation score norm
        """
        trans_0, rot_0 = _extract_trans_rots(rigids_0)

        if not self._diffuse_rot:
            rot_t, rot_score, rot_score_scaling = (
                rot_0,
                np.zeros_like(rot_0),
                np.ones_like(t)
            )
        else:
            rot_t, rot_score = self._so3_diffuser.forward_marginal(rot_0, t)
            rot_score_scaling = self._so3_diffuser.score_scaling(t)

        if not self._diffuse_trans:
            trans_t, trans_score, trans_score_scaling = (
                trans_0,
                np.zeros_like(trans_0),
                np.ones_like(t)
            )
        else:
            trans_t, trans_score = self._r3_diffuser.forward_marginal(trans_0, t)
            trans_score_scaling = self._r3_diffuser.score_scaling(t)

        if diffuse_mask is not None:
            # diffuse_mask = torch.tensor(diffuse_mask).to(rot_t.device)
            rot_t = self._apply_mask(
                rot_t, rot_0, diffuse_mask[..., None])
            trans_t = self._apply_mask(
                trans_t, trans_0, diffuse_mask[..., None])

            trans_score = self._apply_mask(
                trans_score,
                np.zeros_like(trans_score),
                diffuse_mask[..., None])
            rot_score = self._apply_mask(
                rot_score,
                np.zeros_like(rot_score),
                diffuse_mask[..., None])

        if align_trans:
            # make trans_t center first, than align with trans_0; this is not for scaffolding and imputation
            trans_t = trans_t - trans_t.sum(0, keepdims=True) / (diffuse_mask.sum(0, keepdims=True) + 1e-10)
            try:
                trans_t = kabsch_align_np(trans_0, trans_t, diffuse_mask[..., None])
            except:
                trans_t = trans_t
            # Noted that trans_score is not useful

        rigids_t = _assemble_rigid(rot_t, trans_t)
        if as_tensor_7:
            rigids_t = rigids_t.to_tensor_7()
        return {
            'rigids_t': rigids_t,
            'trans_score': trans_score,
            'rot_score': rot_score,
            'trans_score_scaling': trans_score_scaling,
            'rot_score_scaling': rot_score_scaling,
        }

    def calc_trans_0(self, trans_score, trans_t, t):
        return self._r3_diffuser.calc_trans_0(trans_score, trans_t, t)

    def calc_trans_score(self, trans_t, trans_0, t, use_torch=False, scale=True):
        return self._r3_diffuser.score(
            trans_t, trans_0, t, use_torch=use_torch, scale=scale)

    def calc_rot_score(self, rots_t, rots_0, t):
        rots_0_inv = rots_0.invert()
        quats_0_inv = rots_0_inv.get_quats()
        quats_t = rots_t.get_quats()
        quats_0t = ru.quat_multiply(quats_0_inv, quats_t)
        rotvec_0t = du.quat_to_rotvec(quats_0t)
        return self._so3_diffuser.torch_score(rotvec_0t, t)

    def _apply_mask(self, x_diff, x_fixed, diff_mask):
        return diff_mask * x_diff + (1 - diff_mask) * x_fixed

    def trans_parameters(self, trans_t, score_t, t, dt, mask):
        return self._r3_diffuser.distribution(
            trans_t, score_t, t, dt, mask)

    def score(
            self,
            rigid_0: ru.Rigid,
            rigid_t: ru.Rigid,
            t: float):
        tran_0, rot_0 = _extract_trans_rots(rigid_0)
        tran_t, rot_t = _extract_trans_rots(rigid_t)

        if not self._diffuse_rot:
            rot_score = np.zeros_like(rot_0)
        else:
            rot_score = self._so3_diffuser.score(
                rot_t, t)

        if not self._diffuse_trans:
            trans_score = np.zeros_like(tran_0)
        else:
            trans_score = self._r3_diffuser.score(tran_t, tran_0, t)

        return trans_score, rot_score

    def score_scaling(self, t):
        rot_score_scaling = self._so3_diffuser.score_scaling(t)
        trans_score_scaling = self._r3_diffuser.score_scaling(t)
        return rot_score_scaling, trans_score_scaling

    def reverse(
            self,
            rigid_t: ru.Rigid,
            rot_score: np.ndarray,
            trans_score: np.ndarray,
            t: float,
            dt: float,
            diffuse_mask: np.ndarray = None,
            center: bool=True,
            noise_scale: float=1.0,
        ):
        """Reverse sampling function from (t) to (t-1).

        Args:
            rigid_t: [..., N] protein rigid objects at time t.
            rot_score: [..., N, 3] rotation score.
            trans_score: [..., N, 3] translation score.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: [..., N] which residues to update.
            center: true to set center of mass to zero after step

        Returns:
            rigid_t_1: [..., N] protein rigid objects at time t-1.
        """
        trans_t, rot_t = _extract_trans_rots(rigid_t)
        if not self._diffuse_rot:
            rot_t_1 = rot_t
        else:
            rot_t_1 = self._so3_diffuser.reverse(
                rot_t=rot_t,
                score_t=rot_score,
                t=t,
                dt=dt,
                noise_scale=noise_scale,
                )

        # Note: mask is not used in trans reverse; when center is true, it may cause error.
        if not self._diffuse_trans:
            trans_t_1 = trans_t
        else:
            trans_t_1 = self._r3_diffuser.reverse(
                x_t=trans_t,
                score_t=trans_score,
                t=t,
                dt=dt,
                center=center,
                noise_scale=noise_scale
                )

        if diffuse_mask is not None:
            trans_t_1 = self._apply_mask(
                trans_t_1, trans_t, diffuse_mask[..., None])
            rot_t_1 = self._apply_mask(
                rot_t_1, rot_t, diffuse_mask[..., None])

        return _assemble_rigid(rot_t_1, trans_t_1)

    def sample_ref(
            self,
            n_samples: int,
            impute: ru.Rigid=None,
            diffuse_mask: np.ndarray=None,
            as_tensor_7: bool=False
        ):
        """Samples rigids from reference distribution.

        Args:
            n_samples: Number of samples.
            impute: Rigid objects to use as imputation values if either
                translations or rotations are not diffused.
        """
        if impute is not None:
            assert impute.shape[0] == n_samples
            trans_impute, rot_impute = _extract_trans_rots(impute)
            trans_impute = trans_impute.reshape((n_samples, 3))
            rot_impute = rot_impute.reshape((n_samples, 3))
            trans_impute = self._r3_diffuser._scale(trans_impute)

        if diffuse_mask is not None and impute is None:
            raise ValueError('Must provide imputation values.')

        if (not self._diffuse_rot) and impute is None:
            raise ValueError('Must provide imputation values.')

        if (not self._diffuse_trans) and impute is None:
            raise ValueError('Must provide imputation values.')

        if self._diffuse_rot:
            rot_ref = self._so3_diffuser.sample_ref(
                n_samples=n_samples)
        else:
            rot_ref = rot_impute

        if self._diffuse_trans:
            trans_ref = self._r3_diffuser.sample_ref(
                n_samples=n_samples
            )
        else:
            trans_ref = trans_impute

        if diffuse_mask is not None:
            rot_ref = self._apply_mask(
                rot_ref, rot_impute, diffuse_mask[..., None])
            trans_ref = self._apply_mask(
                trans_ref, trans_impute, diffuse_mask[..., None])
        trans_ref = self._r3_diffuser._unscale(trans_ref)
        rigids_t = _assemble_rigid(rot_ref, trans_ref)
        if as_tensor_7:
            rigids_t = rigids_t.to_tensor_7()
        return {'rigids_t': rigids_t}
