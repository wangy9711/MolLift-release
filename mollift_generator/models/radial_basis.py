"""Radial basis functions for spherical edge embeddings."""

from __future__ import annotations

import math

import torch


def reciprocal_bernstein(dist: torch.Tensor, num: int) -> torch.Tensor:
    """Reciprocal Bernstein radial basis (e3x-compatible layout).

    Args:
        dist: distances [...], non-negative
        num: number of basis functions

    Returns:
        Tensor [..., num]
    """
    if num <= 0:
        raise ValueError(f"num must be positive, got {num}")
    r = 1.0 / (1.0 + dist.clamp_min(0.0))
    degree = num - 1
    out = []
    for k in range(num):
        coeff = math.comb(degree, k)
        out.append(coeff * (r**k) * ((1.0 - r) ** (degree - k)))
    return torch.stack(out, dim=-1)
