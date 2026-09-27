"""Evaluation metrics."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from mollift_generator.data.proj_mask import pair_mask_from_proj_mask
from mollift_generator.utils.chirality import chirality_correct  # re-exported; Method A
from mollift_generator.utils.tensor_utils import masked_mean

__all__ = [
    "chirality_correct",
    "kabsch_align",
    "kabsch_rmsd",
    "kabsch_rmsd_reflection_invariant",
    "handedness_correct",
    "projection_rmse",
    "projection_pair_rmse",
    "z_statistics",
    "bond_length_mae",
]


def kabsch_align(
    pred: np.ndarray, truth: np.ndarray, allow_reflection: bool = False
) -> np.ndarray:
    pred_c = pred - pred.mean(axis=0)
    truth_c = truth - truth.mean(axis=0)
    # Row-vector convention: H = U S V^T gives pred_c @ (U V^T).
    # Enforce det(R) = +1 unless reflections are allowed.
    h = pred_c.T @ truth_c
    u, _, vt = np.linalg.svd(h)
    if allow_reflection:
        r = u @ vt
    else:
        d = np.sign(np.linalg.det(u @ vt))
        r = u @ np.diag([1.0, 1.0, d]) @ vt
    return pred_c @ r + truth.mean(axis=0)


def kabsch_rmsd(
    pred: np.ndarray, truth: np.ndarray, allow_reflection: bool = False
) -> float:
    aligned = kabsch_align(pred, truth, allow_reflection=allow_reflection)
    return float(np.sqrt(np.mean(np.sum((aligned - truth) ** 2, axis=1))))


def kabsch_rmsd_reflection_invariant(pred: np.ndarray, truth: np.ndarray) -> float:
    """RMSD under the best orthogonal alignment, including reflections."""
    return kabsch_rmsd(pred, truth, allow_reflection=True)


def handedness_correct(pred: np.ndarray, truth: np.ndarray) -> bool:
    """Whether the best orthogonal alignment is a proper rotation.

    This tests global orientation, not individual stereocenters or geometry quality.
    """
    pred_c = pred - pred.mean(axis=0)
    truth_c = truth - truth.mean(axis=0)
    h = pred_c.T @ truth_c
    u, _, vt = np.linalg.svd(h)
    return bool(np.linalg.det(u @ vt) > 0)


def projection_rmse(
    pred: torch.Tensor,
    pos_2d: torch.Tensor,
    node_mask: torch.Tensor,
    proj_mask: torch.Tensor | None = None,
) -> float:
    mask = proj_mask if proj_mask is not None else node_mask
    if not mask.any():
        return 0.0
    diff = (pred[..., :2] - pos_2d) ** 2
    val = masked_mean(diff.sum(dim=-1), mask)
    return float(torch.sqrt(val).item())


def projection_pair_rmse(
    pred: torch.Tensor,
    pos_2d: torch.Tensor,
    node_mask: torch.Tensor,
    proj_mask: torch.Tensor | None = None,
) -> float:
    mask = proj_mask if proj_mask is not None else node_mask
    pair_mask = pair_mask_from_proj_mask(mask)
    if not pair_mask.any():
        return 0.0
    d_hat = torch.norm(pred[..., :2].unsqueeze(2) - pred[..., :2].unsqueeze(1), dim=-1)
    d_ref = torch.norm(pos_2d.unsqueeze(2) - pos_2d.unsqueeze(1), dim=-1)
    sq = (d_hat - d_ref) ** 2
    return float(torch.sqrt(masked_mean(sq, pair_mask)).item())


def z_statistics(pred: torch.Tensor, node_mask: torch.Tensor) -> Dict[str, float]:
    z = pred[..., 2]
    mask = node_mask
    z_var = masked_mean(z**2, mask) - masked_mean(z, mask) ** 2
    z_abs = masked_mean(torch.abs(z), mask)
    frac = masked_mean((torch.abs(z) > 0.1).to(z.dtype), mask)
    return {
        "z_variance": float(z_var.item()),
        "mean_abs_z": float(z_abs.item()),
        "frac_abs_z_gt_0.1": float(frac.item()),
    }


def bond_length_mae(
    pred: np.ndarray,
    truth: np.ndarray,
    bond_pairs: list[tuple[int, int]],
) -> float:
    errs = []
    for i, j in bond_pairs:
        d_pred = np.linalg.norm(pred[i] - pred[j])
        d_true = np.linalg.norm(truth[i] - truth[j])
        errs.append(abs(d_pred - d_true))
    return float(np.mean(errs)) if errs else 0.0
