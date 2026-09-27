"""Tensor utilities for masked batch operations."""

from __future__ import annotations

import torch


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    keepdim: bool = False,
) -> torch.Tensor:
    """Mean over entries where mask is True."""
    mask_f = mask.to(dtype=values.dtype)
    while mask_f.ndim < values.ndim:
        mask_f = mask_f.unsqueeze(-1)
    num = (values * mask_f).sum(dim=dim, keepdim=keepdim)
    den = mask_f.sum(dim=dim, keepdim=keepdim).clamp_min(1.0)
    return num / den


def center_coords(
    coords: torch.Tensor,
    node_mask: torch.Tensor,
) -> torch.Tensor:
    """Remove masked centroid from coordinates [B, N, 3]."""
    mask = node_mask.unsqueeze(-1).to(dtype=coords.dtype)
    mean = masked_mean(coords, node_mask, dim=1, keepdim=True)
    centered = (coords - mean) * mask
    return centered


def pair_mask_from_node_mask(node_mask: torch.Tensor) -> torch.Tensor:
    return node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
