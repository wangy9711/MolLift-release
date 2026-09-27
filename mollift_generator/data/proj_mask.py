"""Projection observation masks for partial 2D conditioning."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch


def _num_observed(n_atoms: int, ratio_proj: float) -> int:
    if ratio_proj <= 0.0:
        return 0
    if ratio_proj >= 1.0:
        return n_atoms
    return max(1, int(round(n_atoms * ratio_proj)))


def pair_mask_from_proj_mask(proj_mask: torch.Tensor) -> torch.Tensor:
    """True for atom pairs where both endpoints have projection observations."""
    return proj_mask.unsqueeze(-1) & proj_mask.unsqueeze(-2)


def sample_proj_mask_random(
    node_mask: torch.Tensor,
    ratio_proj: float,
) -> torch.Tensor:
    """Per-graph random subset of atoms with 2D projection (training)."""
    proj_mask = torch.zeros_like(node_mask)
    if ratio_proj >= 1.0:
        return node_mask.clone()
    if ratio_proj <= 0.0:
        return proj_mask

    for b in range(node_mask.shape[0]):
        idx = node_mask[b].nonzero(as_tuple=True)[0]
        n = int(idx.numel())
        if n == 0:
            continue
        k = _num_observed(n, ratio_proj)
        perm = torch.randperm(n, device=node_mask.device)
        chosen = idx[perm[:k]]
        proj_mask[b, chosen] = True
    return proj_mask


def sample_proj_mask_random_choices(
    node_mask: torch.Tensor,
    ratio_choices: Sequence[float],
) -> torch.Tensor:
    """Sample one projection-observation ratio independently per graph.

    This is input masking for mixed-ratio training, not neural-network dropout. The
    selected ratio controls the exact rounded number of observed atoms in each graph.
    """
    choices = tuple(float(ratio) for ratio in ratio_choices)
    if not choices:
        raise ValueError("ratio_choices must contain at least one value")
    if any(ratio < 0.0 or ratio > 1.0 for ratio in choices):
        raise ValueError(f"ratio_choices must lie in [0, 1], got {choices}")

    proj_mask = torch.zeros_like(node_mask)
    selected = torch.randint(len(choices), (node_mask.shape[0],), device=node_mask.device)
    for b, choice_idx in enumerate(selected.tolist()):
        idx = node_mask[b].nonzero(as_tuple=True)[0]
        n = int(idx.numel())
        if n == 0:
            continue
        k = _num_observed(n, choices[choice_idx])
        if k == 0:
            continue
        if k == n:
            proj_mask[b, idx] = True
            continue
        perm = torch.randperm(n, device=node_mask.device)
        proj_mask[b, idx[perm[:k]]] = True
    return proj_mask


def sample_proj_mask_deterministic(
    node_mask: torch.Tensor,
    ratio_proj: float,
    keys: Sequence[int],
    seed: int = 42,
) -> torch.Tensor:
    """Reproducible per-graph projection mask (evaluation)."""
    proj_mask = torch.zeros_like(node_mask)
    if ratio_proj >= 1.0:
        return node_mask.clone()
    if ratio_proj <= 0.0:
        return proj_mask

    for b in range(node_mask.shape[0]):
        idx = node_mask[b].nonzero(as_tuple=True)[0]
        n = int(idx.numel())
        if n == 0:
            continue
        k = _num_observed(n, ratio_proj)
        rng = np.random.default_rng(seed + int(keys[b]) * 10007)
        perm = rng.permutation(n)
        chosen = idx[torch.tensor(perm[:k], device=node_mask.device, dtype=torch.long)]
        proj_mask[b, chosen] = True
    return proj_mask


def proj_mask_from_indices(
    n_atoms: int,
    indices: Iterable[int],
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Explicit atom indices with 2D observations (inference)."""
    mask = torch.zeros(n_atoms, dtype=torch.bool, device=device)
    for i in indices:
        idx = int(i)
        if idx < 0 or idx >= n_atoms:
            raise ValueError(f"proj index {idx} out of range for n_atoms={n_atoms}")
        mask[idx] = True
    return mask


def build_pos_2d_with_mask(
    pos_2d_full: np.ndarray,
    proj_mask: torch.Tensor,
) -> np.ndarray:
    """Zero out unobserved atom coordinates before batching."""
    out = np.asarray(pos_2d_full, dtype=np.float32).copy()
    mask = proj_mask.detach().cpu().numpy().astype(bool)
    out[~mask] = 0.0
    return out


def apply_proj_mask_to_batch(
    batch: dict,
    proj_mask: torch.Tensor,
) -> dict:
    """Attach proj_mask and zero unobserved 2D coordinates in a collated batch."""
    out = dict(batch)
    out["proj_mask"] = proj_mask
    out["pos_2d"] = batch["pos_2d"] * proj_mask.unsqueeze(-1).to(batch["pos_2d"].dtype)
    return out
