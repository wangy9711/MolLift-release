"""2D coordinate construction: synthetic projection and RDKit 2D layout."""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

ProjectionMode = Literal["synthetic", "rdkit_2d"]
CoordUnits = Literal["angstrom", "normalized"]


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    u1, u2, u3 = rng.random(3)
    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    return np.array(
        [
            [1 - 2 * (q3**2 + q4**2), 2 * (q2 * q3 - q1 * q4), 2 * (q2 * q4 + q1 * q3)],
            [2 * (q2 * q3 + q1 * q4), 1 - 2 * (q2**2 + q4**2), 2 * (q3 * q4 - q1 * q2)],
            [2 * (q2 * q4 - q1 * q3), 2 * (q3 * q4 + q1 * q2), 1 - 2 * (q2**2 + q3**2)],
        ],
        dtype=np.float64,
    )


def center_coords(coords: np.ndarray) -> np.ndarray:
    return coords - coords.mean(axis=0, keepdims=True)


def normalize_coords_2d(
    pos_2d: np.ndarray,
    scale: float | None = None,
) -> Tuple[np.ndarray, float]:
    """Center 2D coordinates and divide by a supplied or RMS radial scale."""
    pos = center_coords(pos_2d)
    if scale is None:
        scale = float(np.sqrt(np.mean(np.sum(pos**2, axis=-1)).clip(min=1e-8)))
    return pos / scale, scale


def prepare_sample_coords(
    x3d: np.ndarray,
    pos_2d: np.ndarray,
    mode: ProjectionMode,
    coord_units: CoordUnits = "angstrom",
) -> Tuple[np.ndarray, np.ndarray]:
    """Center 3D; synthetic uses xy slice, rdkit_2d keeps Å layout (decentered only)."""
    if coord_units != "angstrom":
        return normalize_sample(x3d, pos_2d, mode)
    x3d = center_coords(x3d)
    if mode == "synthetic":
        pos_2d = x3d[:, :2].copy()
    else:
        pos_2d = center_coords(pos_2d)
    return x3d.astype(np.float32), pos_2d.astype(np.float32)


def normalize_sample(
    x3d: np.ndarray,
    pos_2d: np.ndarray,
    mode: ProjectionMode,
) -> Tuple[np.ndarray, np.ndarray]:
    """Center coordinates and normalize the projection's RMS radius to one."""
    x3d = center_coords(x3d)
    if mode == "synthetic":
        scale = float(np.sqrt(np.mean(np.sum(x3d[:, :2] ** 2, axis=-1)).clip(min=1e-8)))
        x3d = x3d / scale
        pos_2d = x3d[:, :2].copy()
    else:
        pos_2d, _ = normalize_coords_2d(pos_2d)
    return x3d.astype(np.float32), pos_2d.astype(np.float32)


def synthetic_projection(
    x3d: np.ndarray,
    rng: np.random.Generator | None = None,
    coord_units: CoordUnits = "angstrom",
) -> Tuple[np.ndarray, np.ndarray]:
    """Center, random SO(3) rotate, take xy as 2D projection (Å by default)."""
    rng = rng or np.random.default_rng()
    x3d_c = center_coords(x3d.astype(np.float64))
    rot = _random_rotation_matrix(rng)
    x3d_rot = x3d_c @ rot.T
    pos_2d = x3d_rot[:, :2].copy()
    return prepare_sample_coords(x3d_rot, pos_2d, mode="synthetic", coord_units=coord_units)


def rdkit_2d_coords(mol: Chem.Mol) -> np.ndarray:
    mol_copy = Chem.Mol(mol)
    AllChem.Compute2DCoords(mol_copy)
    conf = mol_copy.GetConformer()
    n = mol_copy.GetNumAtoms()
    return np.array(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in range(n)],
        dtype=np.float64,
    )


def build_2d_coords(
    x3d: np.ndarray,
    mol: Chem.Mol,
    mode: ProjectionMode,
    rng: np.random.Generator | None = None,
    coord_units: CoordUnits = "angstrom",
) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "synthetic":
        return synthetic_projection(x3d, rng=rng, coord_units=coord_units)
    pos_2d = rdkit_2d_coords(mol)
    return prepare_sample_coords(x3d, pos_2d, mode="rdkit_2d", coord_units=coord_units)
