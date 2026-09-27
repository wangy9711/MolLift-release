"""Initial 3D coordinates from 2D observations and RDKit embed."""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from mollift_solver.projection import Plane2DFrame, lift_2d_to_3d_numpy, project_3d_to_2d_numpy
from mollift_solver.topology import get_coords_from_mol
from mollift_solver.types import SolveConfig


def _resolve_reproj_mask(
    coords_2d: np.ndarray,
    coords_2d_mask: Optional[np.ndarray],
) -> np.ndarray:
    """Return bool mask (N,) for reproj loss; infer from NaN if mask not given."""
    coords_2d = np.asarray(coords_2d, dtype=np.float64)
    if coords_2d_mask is not None:
        return np.asarray(coords_2d_mask, dtype=bool)
    if np.any(np.isnan(coords_2d)):
        return ~np.any(np.isnan(coords_2d), axis=1)
    return np.ones(coords_2d.shape[0], dtype=bool)


def rdkit_depiction_2d(mol: Chem.Mol) -> np.ndarray:
    """RDKit 2D layout coordinates (N, 2) in arbitrary depiction units."""
    mol_copy = Chem.Mol(mol)
    AllChem.Compute2DCoords(mol_copy)
    conf = mol_copy.GetConformer()
    n = mol_copy.GetNumAtoms()
    return np.array(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in range(n)],
        dtype=np.float64,
    )


def align_layout_to_observed(
    layout_2d: np.ndarray,
    observed_2d: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Similarity-transform a 2D layout to match observed annotated atoms."""
    layout_2d = np.asarray(layout_2d, dtype=np.float64)
    observed_2d = np.asarray(observed_2d, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    src = layout_2d[mask]
    tgt = observed_2d[mask]
    src_cent = src.mean(axis=0)
    tgt_cent = tgt.mean(axis=0)
    src_c = src - src_cent
    tgt_c = tgt - tgt_cent

    h = src_c.T @ tgt_c
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T

    denom = float(np.trace(src_c.T @ src_c))
    scale = float(np.trace(tgt_c.T @ (src_c @ r)) / denom) if denom > 1e-12 else 1.0

    layout_c = layout_2d - src_cent
    return scale * (layout_c @ r) + tgt_cent


def fill_missing_2d(
    coords_2d: np.ndarray,
    mask: np.ndarray,
    mol: Chem.Mol,
    frame: Plane2DFrame,
    random_seed: int = 42,
) -> np.ndarray:
    """Fill unobserved 2D rows from embed projection or aligned 2D depiction."""
    coords_2d = np.asarray(coords_2d, dtype=np.float64).copy()
    mask = np.asarray(mask, dtype=bool)
    if mask.all():
        return coords_2d

    proj: np.ndarray | None = None
    for seed in range(random_seed, random_seed + 8):
        embedded = rdkit_embed_coords(mol, random_seed=seed)
        if embedded is not None:
            proj = project_3d_to_2d_numpy(embedded, frame)
            break

    if proj is None:
        layout = rdkit_depiction_2d(mol)
        proj = align_layout_to_observed(layout, coords_2d, mask)

    missing = ~mask
    coords_2d[missing] = proj[missing]
    return coords_2d


def kabsch_align(
    source: np.ndarray,
    target: np.ndarray,
    allow_scale: bool = False,
) -> np.ndarray:
    """
    Align source (N, 3) to target (N, 3) via optimal rotation + translation.
    Returns aligned source.
    """
    src = np.asarray(source, dtype=float)
    tgt = np.asarray(target, dtype=float)
    src_cent = src.mean(axis=0)
    tgt_cent = tgt.mean(axis=0)
    src_c = src - src_cent
    tgt_c = tgt - tgt_cent

    h = src_c.T @ tgt_c
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T

    if allow_scale:
        scale = np.trace(tgt_c.T @ (src_c @ r)) / np.trace(src_c.T @ src_c)
    else:
        scale = 1.0

    return scale * (src_c @ r) + tgt_cent


def lift_2d_with_depth(
    coords_2d: np.ndarray,
    frame: Plane2DFrame,
    depth: float = 0.0,
) -> np.ndarray:
    base = lift_2d_to_3d_numpy(coords_2d, frame)
    return base + depth * frame.normal


def rdkit_embed_coords(
    mol: Chem.Mol,
    random_seed: int = 42,
) -> Optional[np.ndarray]:
    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)

    # Explicit hydrogens make ETKDG geometry and stereochemistry more reliable and
    # avoid RDKit's no-explicit-H warning. Retain the heavy-only path as fallback.
    for use_hs in (True, False):
        mol_work = Chem.AddHs(Chem.Mol(mol)) if use_hs else Chem.Mol(mol)
        if AllChem.EmbedMolecule(mol_work, params) != 0:
            continue
        if AllChem.MMFFHasAllMoleculeParams(mol_work):
            AllChem.MMFFOptimizeMolecule(mol_work, mmffVariant="MMFF94s", maxIters=200)
        elif AllChem.UFFHasAllMoleculeParams(mol_work):
            AllChem.UFFOptimizeMolecule(mol_work, maxIters=200)
        if use_hs:
            mol_work = Chem.RemoveHs(mol_work)
        if mol_work.GetNumAtoms() != mol.GetNumAtoms():
            continue
        return get_coords_from_mol(mol_work)
    return None


def align_embed_to_2d(
    coords_3d: np.ndarray,
    coords_2d: np.ndarray,
    frame: Plane2DFrame,
) -> np.ndarray:
    """Rigidly orient a 3D conformer to minimize its orthographic 2D mismatch.

    This is a rectangular orthogonal-Procrustes fit from centered 3D coordinates
    to the two observed in-plane axes. Unlike alignment to a flat 3D scaffold, it
    preserves the conformer's internal geometry and depth variation.
    """
    coords_3d = np.asarray(coords_3d, dtype=np.float64)
    coords_2d = np.asarray(coords_2d, dtype=np.float64)
    x = coords_3d - coords_3d.mean(axis=0, keepdims=True)
    y_center = coords_2d.mean(axis=0)
    y = coords_2d - y_center
    u, _, vt = np.linalg.svd(x.T @ y, full_matrices=False)
    q2 = u @ vt  # (3, 2), orthonormal columns
    q0, q1 = q2[:, 0], q2[:, 1]
    qn = np.cross(q0, q1)
    qn /= max(np.linalg.norm(qn), 1e-12)
    local = np.stack([x @ q0, x @ q1, x @ qn], axis=1)
    return (
        frame.origin
        + (local[:, 0:1] + y_center[0]) * frame.u_axis
        + (local[:, 1:2] + y_center[1]) * frame.v_axis
        + local[:, 2:3] * frame.normal
    )


def reflect_across_frame(coords_3d: np.ndarray, frame: Plane2DFrame) -> np.ndarray:
    """Reflect depth while leaving the orthographic projection unchanged."""
    coords = np.asarray(coords_3d, dtype=np.float64)
    depth = ((coords - frame.origin) * frame.normal).sum(axis=1, keepdims=True)
    return coords - 2.0 * depth * frame.normal


def stochastic_depth_lift(
    coords_2d: np.ndarray,
    mol: Chem.Mol,
    frame: Plane2DFrame,
    rng: np.random.Generator,
    depth_scale: float,
    smoothing_steps: int,
) -> np.ndarray:
    """Non-planar graph-smoothed lift that avoids the flat z-gradient symmetry."""
    n = mol.GetNumAtoms()
    depth = rng.normal(0.0, max(depth_scale, 0.05), size=n)
    neighbors = [[a.GetIdx() for a in mol.GetAtomWithIdx(i).GetNeighbors()] for i in range(n)]
    for _ in range(max(0, smoothing_steps)):
        updated = depth.copy()
        for i, ns in enumerate(neighbors):
            if ns:
                updated[i] = 0.55 * depth[i] + 0.45 * float(np.mean(depth[ns]))
        depth = updated
    depth -= depth.mean()
    return lift_2d_to_3d_numpy(coords_2d, frame) + depth[:, None] * frame.normal


def generate_initial_coords(
    mol: Chem.Mol,
    coords_2d: np.ndarray,
    config: SolveConfig,
    frame: Plane2DFrame,
    coords_3d_init: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    coords_2d_mask: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """Build list of initial (N, 3) arrays for multi-start optimization."""
    coords_2d = np.asarray(coords_2d, dtype=np.float64)
    n = mol.GetNumAtoms()
    mask = _resolve_reproj_mask(coords_2d, coords_2d_mask)
    coords_2d_full = fill_missing_2d(
        coords_2d,
        mask,
        mol,
        frame,
        random_seed=config.embed_random_seed_base,
    )

    if coords_3d_init is not None:
        if isinstance(coords_3d_init, list):
            inits = [np.asarray(x, dtype=np.float64).reshape(n, 3) for x in coords_3d_init]
        else:
            inits = [np.asarray(coords_3d_init, dtype=np.float64).reshape(n, 3)]
        aligned = [align_embed_to_2d(x, coords_2d_full, frame) for x in inits]
        if config.include_mirror_starts:
            aligned.extend(reflect_across_frame(x, frame) for x in list(aligned))
        return aligned

    inits: List[np.ndarray] = []
    rng = np.random.default_rng(config.embed_random_seed_base)

    base_count = config.n_starts
    if config.include_mirror_starts and config.n_starts > 1:
        # Reserve one slot for the unavoidable orthographic depth mirror, while
        # keeping the remaining slots as genuinely different conformer basins.
        base_count = config.n_starts - 1

    for s in range(base_count):
        seed = config.embed_random_seed_base + s
        embedded = rdkit_embed_coords(mol, random_seed=seed)
        if embedded is not None and not (base_count >= 3 and s == base_count - 1):
            x0 = align_embed_to_2d(embedded, coords_2d_full, frame)
        else:
            jitter = rng.normal(0, 0.015, size=(n, 2))
            x0 = stochastic_depth_lift(
                coords_2d_full + jitter,
                mol,
                frame,
                rng,
                depth_scale=config.depth_noise,
                smoothing_steps=config.init_depth_smoothing_steps,
            )
        inits.append(x0.astype(np.float32))

    if config.include_mirror_starts:
        if inits and len(inits) < config.n_starts:
            inits.append(reflect_across_frame(inits[0], frame).astype(np.float32))

    return inits[: config.n_starts]
