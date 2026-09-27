"""Vectorized differentiable losses for constrained embedding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from mollift_solver.projection import Plane2DFrame, project_3d_to_2d_torch


@dataclass
class LossBreakdown:
    """Per-structure losses; tensors are scalar or have shape ``(B,)``."""

    total: torch.Tensor
    reproj: torch.Tensor
    bond: torch.Tensor
    angle: torch.Tensor
    torsion: torch.Tensor
    chirality: torch.Tensor
    planarity: torch.Tensor
    clash: torch.Tensor


def _batched(coords: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if coords.ndim == 2:
        return coords.unsqueeze(0), True
    if coords.ndim != 3:
        raise ValueError(f"coords must be (N, 3) or (B, N, 3), got {tuple(coords.shape)}")
    return coords, False


def _restore(values: torch.Tensor, squeeze: bool) -> torch.Tensor:
    return values[0] if squeeze else values


def _zero(coords: torch.Tensor, squeeze: bool) -> torch.Tensor:
    values = coords.new_zeros(coords.shape[0])
    return _restore(values, squeeze)


def bond_length_loss(coords: torch.Tensor, edges: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    xyz, squeeze = _batched(coords)
    if edges.numel() == 0:
        return _zero(xyz, squeeze)
    distances = torch.linalg.vector_norm(xyz[:, edges[:, 0]] - xyz[:, edges[:, 1]], dim=-1)
    return _restore(((distances - targets) ** 2).mean(dim=1), squeeze)


def angle_loss(coords: torch.Tensor, triplets: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    xyz, squeeze = _batched(coords)
    if triplets.numel() == 0:
        return _zero(xyz, squeeze)
    v1 = xyz[:, triplets[:, 0]] - xyz[:, triplets[:, 1]]
    v2 = xyz[:, triplets[:, 2]] - xyz[:, triplets[:, 1]]
    denom = (torch.linalg.vector_norm(v1, dim=-1) * torch.linalg.vector_norm(v2, dim=-1)).clamp_min(1e-8)
    cosine = (v1 * v2).sum(dim=-1) / denom
    angles = torch.acos(cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
    return _restore(((angles - targets) ** 2).mean(dim=1), squeeze)


def torsion_loss(coords: torch.Tensor, quads: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """MMFF94 periodic torsion energy."""
    xyz, squeeze = _batched(coords)
    if quads.numel() == 0:
        return _zero(xyz, squeeze)
    b0 = xyz[:, quads[:, 0]] - xyz[:, quads[:, 1]]
    b1 = xyz[:, quads[:, 2]] - xyz[:, quads[:, 1]]
    b2 = xyz[:, quads[:, 3]] - xyz[:, quads[:, 2]]
    b1n = b1 / torch.linalg.vector_norm(b1, dim=-1, keepdim=True).clamp_min(1e-8)
    v = b0 - (b0 * b1n).sum(dim=-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(dim=-1, keepdim=True) * b1n
    phi = torch.atan2((torch.cross(b1n, v, dim=-1) * w).sum(dim=-1), (v * w).sum(dim=-1))
    v1, v2, v3 = params.unbind(dim=1)
    energy = 0.5 * (v1 * (1 + torch.cos(phi)) + v2 * (1 - torch.cos(2 * phi)) + v3 * (1 + torch.cos(3 * phi)))
    return _restore(energy.mean(dim=1), squeeze)


def chirality_loss(
    coords: torch.Tensor, centers: torch.Tensor, target_signs: torch.Tensor, margin: float = 0.08
) -> torch.Tensor:
    xyz, squeeze = _batched(coords)
    if centers.numel() == 0:
        return _zero(xyz, squeeze)
    va = xyz[:, centers[:, 0]] - xyz[:, centers[:, 1]]
    vb = xyz[:, centers[:, 2]] - xyz[:, centers[:, 1]]
    vc = xyz[:, centers[:, 3]] - xyz[:, centers[:, 1]]
    denom = (
        torch.linalg.vector_norm(va, dim=-1)
        * torch.linalg.vector_norm(vb, dim=-1)
        * torch.linalg.vector_norm(vc, dim=-1)
    ).clamp_min(1e-8)
    signed = (torch.cross(va, vb, dim=-1) * vc).sum(dim=-1) / denom
    penalties = torch.relu(margin - target_signs * signed) ** 2
    return _restore(penalties.mean(dim=1) + penalties.max(dim=1).values, squeeze)


def planarity_loss(
    coords: torch.Tensor, group_indices: torch.Tensor, group_mask: torch.Tensor
) -> torch.Tensor:
    """Best-fit-plane loss for padded planar groups, vectorized over starts/groups."""
    xyz, squeeze = _batched(coords)
    if group_indices.numel() == 0:
        return _zero(xyz, squeeze)
    points = xyz[:, group_indices.clamp_min(0)]
    mask = group_mask[None, :, :, None]
    counts = group_mask.sum(dim=1).clamp_min(1).to(xyz.dtype)
    means = (points * mask).sum(dim=2) / counts[None, :, None]
    centered = (points - means[:, :, None]) * mask
    covariance = torch.einsum("bgni,bgnj->bgij", centered, centered) / counts[None, :, None, None]
    terms = torch.linalg.eigvalsh(covariance)[..., 0].clamp_min(0.0)
    return _restore(terms.sum(dim=1), squeeze)


def steric_clash_loss(coords: torch.Tensor, pairs: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
    xyz, squeeze = _batched(coords)
    if not isinstance(pairs, torch.Tensor):
        pairs = torch.as_tensor(pairs, device=xyz.device, dtype=torch.long).reshape(-1, 2)
    if pairs.numel() == 0:
        return _zero(xyz, squeeze)
    distances = torch.linalg.vector_norm(xyz[:, pairs[:, 0]] - xyz[:, pairs[:, 1]], dim=-1)
    penalties = torch.relu(limits - distances) ** 2
    return _restore(penalties.mean(dim=1) + penalties.max(dim=1).values, squeeze)


def reproj_loss(
    coords: torch.Tensor,
    target: torch.Tensor,
    frame: Plane2DFrame,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    xyz, squeeze = _batched(coords)
    pred = project_3d_to_2d_torch(xyz, frame)
    sq = (pred - target) ** 2
    if mask is None:
        values = sq.mean(dim=(-1, -2))
    elif not bool(mask.any()):
        values = xyz.new_zeros(xyz.shape[0])
    else:
        values = sq[:, mask].mean(dim=(-1, -2))
    return _restore(values, squeeze)


def total_loss(
    coords: torch.Tensor,
    coords_2d_target: torch.Tensor,
    frame: Plane2DFrame,
    bond_edges: torch.Tensor,
    target_bond_lengths: torch.Tensor,
    angle_triplets: torch.Tensor,
    target_angles: torch.Tensor,
    torsion_quads: torch.Tensor,
    torsion_params: torch.Tensor,
    chiral_centers: torch.Tensor,
    chiral_signs: torch.Tensor,
    planar_group_indices: torch.Tensor,
    planar_group_mask: torch.Tensor,
    nonbonded_pairs: torch.Tensor,
    clash_distances: torch.Tensor,
    w_reproj: float,
    w_bond: float,
    w_angle: float,
    w_torsion: float,
    w_chirality: float,
    w_planarity: float,
    w_clash: float,
    reproj_mask: Optional[torch.Tensor] = None,
) -> LossBreakdown:
    terms = dict(
        reproj=reproj_loss(coords, coords_2d_target, frame, reproj_mask),
        bond=bond_length_loss(coords, bond_edges, target_bond_lengths),
        angle=angle_loss(coords, angle_triplets, target_angles),
        torsion=torsion_loss(coords, torsion_quads, torsion_params),
        chirality=chirality_loss(coords, chiral_centers, chiral_signs),
        planarity=planarity_loss(coords, planar_group_indices, planar_group_mask),
        clash=steric_clash_loss(coords, nonbonded_pairs, clash_distances),
    )
    total = sum(
        weight * terms[name]
        for name, weight in zip(
            terms,
            (w_reproj, w_bond, w_angle, w_torsion, w_chirality, w_planarity, w_clash),
        )
    )
    return LossBreakdown(total=total, **terms)
