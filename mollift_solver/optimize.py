"""Solver with MMFF torsions and final-objective best-state tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from mollift_solver.initialize import _resolve_reproj_mask, fill_missing_2d
from mollift_solver.losses import LossBreakdown
from mollift_solver import losses as base_losses
from mollift_solver.projection import Plane2DFrame
from mollift_solver.types import MoleculeTopology, SolveConfig, SolveResult


torsion_loss = base_losses.torsion_loss


def total_loss(coords, target, frame, bonds, bond_targets, angles, angle_targets,
               quads, params, centers, signs, planar, planar_mask, pairs, limits,
               *weights, reproj_mask=None):
    torsion = torsion_loss
    terms = dict(
        reproj=base_losses.reproj_loss(coords, target, frame, reproj_mask),
        bond=base_losses.bond_length_loss(coords, bonds, bond_targets),
        angle=base_losses.angle_loss(coords, angles, angle_targets),
        torsion=torsion(coords, quads, params),
        chirality=base_losses.chirality_loss(coords, centers, signs),
        planarity=base_losses.planarity_loss(coords, planar, planar_mask),
        clash=base_losses.steric_clash_loss(coords, pairs, limits),
    )
    return LossBreakdown(total=sum(w * term for w, term in zip(weights, terms.values())), **terms)


@dataclass
class _TopologyTensors:
    bond_edges: torch.Tensor
    target_bonds: torch.Tensor
    angle_triplets: torch.Tensor
    target_angles: torch.Tensor
    torsion_quads: torch.Tensor
    torsion_params: torch.Tensor
    chiral_centers: torch.Tensor
    chiral_signs: torch.Tensor
    planar_indices: torch.Tensor
    planar_mask: torch.Tensor
    nonbonded_pairs: torch.Tensor
    clash_distances: torch.Tensor


def _get_dtype(config: SolveConfig) -> torch.dtype:
    return torch.float32 if config.dtype == "float32" else torch.float64


def _index_tensor(values, width: int, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(values, device=device, dtype=torch.long).reshape(-1, width)


def _topology_tensors(
    topology: MoleculeTopology, device: torch.device, dtype: torch.dtype
) -> _TopologyTensors:
    groups = topology.planar_groups
    max_size = max((len(group) for group in groups), default=0)
    planar_indices = torch.full((len(groups), max_size), -1, device=device, dtype=torch.long)
    planar_mask = torch.zeros((len(groups), max_size), device=device, dtype=torch.bool)
    for row, group in enumerate(groups):
        planar_indices[row, : len(group)] = torch.as_tensor(group, device=device)
        planar_mask[row, : len(group)] = True
    return _TopologyTensors(
        bond_edges=_index_tensor(topology.bond_edges, 2, device),
        target_bonds=torch.as_tensor(topology.target_bond_lengths, device=device, dtype=dtype),
        angle_triplets=_index_tensor(topology.angle_triplets, 3, device),
        target_angles=torch.as_tensor(topology.target_angles, device=device, dtype=dtype),
        torsion_quads=_index_tensor(topology.torsion_quads, 4, device),
        torsion_params=torch.as_tensor(topology.torsion_params, device=device, dtype=dtype).reshape(-1, 3),
        chiral_centers=_index_tensor(topology.chiral_centers, 4, device),
        chiral_signs=torch.as_tensor(topology.chiral_signs, device=device, dtype=dtype),
        planar_indices=planar_indices,
        planar_mask=planar_mask,
        nonbonded_pairs=_index_tensor(topology.nonbonded_pairs, 2, device),
        clash_distances=torch.as_tensor(topology.clash_distances, device=device, dtype=dtype),
    )


def _stage_weights(step: int, config: SolveConfig) -> Tuple[float, ...]:
    reproj = config.w_reproj
    if step < int(config.steps * config.stage1_fraction):
        reproj *= config.stage1_reproj_scale
    return (
        reproj,
        config.w_bond,
        config.w_angle,
        config.w_torsion,
        config.w_chirality,
        config.w_planarity,
        config.w_clash,
    )


def optimize_multi_start(
    inits: List[np.ndarray],
    coords_2d: np.ndarray,
    topology: MoleculeTopology,
    frame: Plane2DFrame,
    config: SolveConfig,
    reproj_mask: Optional[np.ndarray] = None,
) -> SolveResult:
    """Optimize all starts as a single ``(S, N, 3)`` tensor on CPU or GPU."""
    min_stage2_steps = 30
    if config.optimizer.lower() != 'adam':
        raise ValueError('The Solver uses Adam')
    if config.steps - int(config.steps * config.stage1_fraction) < min_stage2_steps:
        raise ValueError('Insufficient scheduled full-weight window')
    if not inits:
        raise ValueError("At least one initial conformer is required")
    device, dtype = torch.device(config.device), _get_dtype(config)
    mask = _resolve_reproj_mask(coords_2d, reproj_mask)
    coords_2d_full = fill_missing_2d(
        coords_2d, mask, topology.mol, frame, random_seed=config.embed_random_seed_base
    )
    target_2d = torch.as_tensor(coords_2d_full, device=device, dtype=dtype)
    mask_t = torch.as_tensor(mask, device=device, dtype=torch.bool)
    topo = _topology_tensors(topology, device, dtype)
    coords = nn.Parameter(torch.as_tensor(np.stack(inits), device=device, dtype=dtype))

    optimizer = torch.optim.Adam([coords], lr=config.lr)

    def evaluate(current: torch.Tensor, weights: Tuple[float, ...]) -> LossBreakdown:
        return total_loss(
            current,
            target_2d,
            frame,
            topo.bond_edges,
            topo.target_bonds,
            topo.angle_triplets,
            topo.target_angles,
            topo.torsion_quads,
            topo.torsion_params,
            topo.chiral_centers,
            topo.chiral_signs,
            topo.planar_indices,
            topo.planar_mask,
            topo.nonbonded_pairs,
            topo.clash_distances,
            *weights,
            reproj_mask=mask_t,
        )

    start_count = len(inits)
    best_losses = torch.full((start_count,), float("inf"), device=device, dtype=dtype)
    best_coords = coords.detach().clone()
    stale = torch.zeros(start_count, device=device, dtype=torch.long)
    histories: List[List[float]] = [[] for _ in inits]

    returned_steps = torch.full((start_count,), -1, device=device, dtype=torch.long)
    full_histories = []
    boundary = int(config.steps * config.stage1_fraction)
    final_weights = _stage_weights(config.steps, config)
    for step in range(config.steps):
        if step == boundary:
            stale.zero_()
        weights = _stage_weights(step, config)
        optimizer.zero_grad()
        evaluate(coords, weights).total.sum().backward()
        optimizer.step()

        with torch.no_grad():
            breakdown = evaluate(coords, weights)
            scheduled = breakdown.total
            full_current = scheduled + (config.w_reproj - weights[0]) * breakdown.reproj
            full_histories.append(full_current.detach().cpu().tolist())
            current = full_current
            improved = current < best_losses
            best_losses = torch.where(improved, current, best_losses)
            best_coords[improved] = coords.detach()[improved]
            returned_steps[improved] = step
            stale = torch.where(improved, torch.zeros_like(stale), stale + 1)
            for idx, value in enumerate(current.detach().cpu().tolist()):
                histories[idx].append(float(value))

        # Batched optimization stops only when every start has stalled.
        can_stop = step + 1 >= boundary + min_stage2_steps
        if can_stop and bool((stale >= config.early_stop_patience).all()) and len(histories[0]) >= 2:
            relative = [
                abs(h[-1] - h[-2]) / (abs(h[-2]) + 1e-12) for h in histories
            ]
            if max(relative) < config.early_stop_tol:
                break

    final_weights = (
        config.w_reproj, config.w_bond, config.w_angle, config.w_torsion,
        config.w_chirality, config.w_planarity, config.w_clash,
    )
    with torch.no_grad():
        final = evaluate(best_coords, final_weights)
        winner = int(torch.argmin(final.total).item())

    candidate_coords = best_coords.cpu().numpy().astype(np.float64)
    candidate_losses = final.total.detach().cpu().numpy().astype(np.float64)

    def value(term: torch.Tensor) -> float:
        return float(term[winner].detach().cpu())

    result = SolveResult(
        coords_3d=best_coords[winner].cpu().numpy().astype(np.float64),
        final_loss=value(final.total),
        loss_reproj=value(final.reproj),
        loss_bond=value(final.bond),
        loss_angle=value(final.angle),
        best_start_idx=winner,
        loss_torsion=value(final.torsion),
        loss_chirality=value(final.chirality),
        loss_planarity=value(final.planarity),
        loss_clash=value(final.clash),
        loss_history=histories[winner],
        frame=frame,
        mol=topology.mol,
        smiles=topology.smiles,
        candidate_coords_3d=candidate_coords,
        candidate_final_losses=candidate_losses,
    )

    diagnostic = {
        "actual_steps": step + 1, "entered_stage2": step >= boundary,
        "returned_steps_zero_based": returned_steps.cpu().tolist(),
        "returned_stage1_count": int((returned_steps < boundary).sum().item()),
        "full_objective_trajectory": full_histories,
        "tracking_objective_trajectory": histories,
        "returned_terms": {key: getattr(final, key).detach().cpu().tolist()
                           for key in ("total", "reproj", "bond", "angle", "torsion",
                                       "chirality", "planarity", "clash")},
    }
    result.diagnostics = diagnostic
    return result
