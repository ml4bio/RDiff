"""The built-in assessor for per-residue lDDT scores."""

import torch
import pdb

def softmax_cross_entropy(logits, labels):
    loss = -1 * torch.sum(
        labels * torch.nn.functional.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss


def permute_final_dims(tensor: torch.Tensor, inds):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def lddt(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 30.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> torch.Tensor:
    n = all_atom_mask.shape[-2]
    dmat_true = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_positions[..., None, :]
                - all_atom_positions[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    dmat_pred = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_pred_pos[..., None, :]
                - all_atom_pred_pos[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )
    dists_to_score = (
        (dmat_true < cutoff)
        * all_atom_mask
        * permute_final_dims(all_atom_mask, (1, 0))
        * (1.0 - torch.eye(n, device=all_atom_mask.device))
    )

    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25

    dims = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    score = norm * (eps + torch.sum(dists_to_score * score, dim=dims))

    return score


def lddt_loss(
    logits: torch.Tensor,
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    no_bins: int = 50,
    eps: float = 1e-10,
    **kwargs,
) -> torch.Tensor:

    n = all_atom_mask.shape[-2]

    c1_pos = 1 # the number of C1' atoms in the all_atom_pred_pos tensor
    all_atom_pred_pos = all_atom_pred_pos[..., c1_pos, :]
    all_atom_positions = all_atom_positions[..., c1_pos, :]
    all_atom_mask = all_atom_mask[..., c1_pos: (c1_pos + 1)]  # keep dim

    score = lddt(
        all_atom_pred_pos,
        all_atom_positions,
        all_atom_mask,
        cutoff=cutoff,
        eps=eps
    )

    # TODO: Remove after initial pipeline testing
    score = torch.nan_to_num(score, nan=torch.nanmean(score))
    score[score < 0] = 0

    score = score.detach()
    bin_index = torch.floor(score * no_bins).long()
    bin_index = torch.clamp(bin_index, max=(no_bins - 1))
    lddt_c1_one_hot = torch.nn.functional.one_hot(
        bin_index, num_classes=no_bins
    )

    errors = softmax_cross_entropy(logits, lddt_c1_one_hot)
    all_atom_mask = all_atom_mask.squeeze(-1)
    loss = torch.sum(errors * all_atom_mask, dim=-1) / (
        eps + torch.sum(all_atom_mask, dim=-1)
    )

    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss
