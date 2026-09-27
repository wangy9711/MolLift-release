"""Pixel coordinates -> solver plane coordinates with automatic scale."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from mollift_solver.types import MoleculeTopology


@dataclass
class CalibrationResult:
    coords_2d: np.ndarray  # (M, 2), NaN for unannotated
    mask: np.ndarray  # (M,) bool
    pixels_per_angstrom: float
    n_bond_pairs_used: int


def estimate_pixels_per_angstrom(
    pixels: np.ndarray,
    mask: np.ndarray,
    topology: MoleculeTopology,
) -> Tuple[float, int]:
    """
    Estimate pixels/Å from annotated bond pairs.

    For each bond (i, j) with both atoms annotated:
        ppa_ij = pixel_distance / target_bond_length
    Returns median ppa and count of pairs used.
    """
    pixels = np.asarray(pixels, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    ratios: list[float] = []

    for edge_idx, (i, j) in enumerate(topology.bond_edges):
        if not (mask[i] and mask[j]):
            continue
        d_px = float(np.linalg.norm(pixels[i] - pixels[j]))
        if d_px < 1e-6:
            continue
        target_len = float(topology.target_bond_lengths[edge_idx])
        ratios.append(d_px / target_len)

    if not ratios:
        raise ValueError(
            "Cannot auto-estimate pixels_per_angstrom: no annotated bond pairs found. "
            "Annotate at least one bonded atom pair, or pass --pixels-per-angstrom manually."
        )

    return float(np.median(ratios)), len(ratios)


def pixels_to_plane(
    pixels: np.ndarray,
    mask: np.ndarray,
    pixels_per_angstrom: float,
) -> np.ndarray:
    """Convert image pixels to solver plane coords (centroid origin, y flipped)."""
    pixels = np.asarray(pixels, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if pixels_per_angstrom <= 0:
        raise ValueError(f"pixels_per_angstrom must be positive, got {pixels_per_angstrom}")

    coords = np.full_like(pixels, np.nan)
    if not mask.any():
        return coords

    annotated = pixels[mask]
    cx = float(annotated[:, 0].mean())
    cy = float(annotated[:, 1].mean())
    ppa = float(pixels_per_angstrom)

    coords[mask, 0] = (pixels[mask, 0] - cx) / ppa
    coords[mask, 1] = -(pixels[mask, 1] - cy) / ppa
    return coords


def calibrate_pixels_to_plane(
    pixels: np.ndarray,
    mask: np.ndarray,
    topology: MoleculeTopology,
    pixels_per_angstrom: Optional[float] = None,
) -> CalibrationResult:
    """Full calibration: estimate or use given ppa, return plane coords."""
    if pixels_per_angstrom is None:
        ppa, n_pairs = estimate_pixels_per_angstrom(pixels, mask, topology)
    else:
        ppa = float(pixels_per_angstrom)
        n_pairs = 0

    coords_2d = pixels_to_plane(pixels, mask, ppa)
    return CalibrationResult(
        coords_2d=coords_2d,
        mask=np.asarray(mask, dtype=bool),
        pixels_per_angstrom=ppa,
        n_bond_pairs_used=n_pairs,
    )
