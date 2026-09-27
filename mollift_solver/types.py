"""Configuration and result types for MolLift_solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np

from mollift_solver.projection import Plane2DFrame


@dataclass
class SolveConfig:
    """Hyperparameters for constrained 2D -> 3D optimization."""

    # Loss weights (stage 2 uses these; stage 1 scales reproj down)
    w_reproj: float = 20.0
    w_bond: float = 10.0
    w_angle: float = 5.0
    w_torsion: float = 0.15
    w_chirality: float = 5.0
    w_planarity: float = 10.0
    w_clash: float = 2.0

    # Stage 1: first fraction of steps emphasizes geometry
    stage1_fraction: float = 0.3
    stage1_reproj_scale: float = 0.1

    # Optimizer
    optimizer: str = "adam"
    lr: float = 0.035
    steps: int = 600
    early_stop_tol: float = 1e-6
    early_stop_patience: int = 30

    # Multi-start (when coords_3d_init is None)
    n_starts: int = 3
    default_depth: float = 0.5
    depth_noise: float = 0.3
    include_mirror_starts: bool = False
    init_depth_smoothing_steps: int = 3

    # Projection frame (None -> default XY plane)
    plane_frame: Optional[Plane2DFrame] = None

    # RDKit embed
    embed_random_seed_base: int = 42

    device: str = "cpu"
    dtype: str = "float32"


@dataclass
class MoleculeTopology:
    """Bond/angle constraints derived from RDKit mol."""

    mol: Any  # rdkit.Chem.Mol
    smiles: str
    bond_edges: List[Tuple[int, int]]
    angle_triplets: List[Tuple[int, int, int]]
    target_bond_lengths: np.ndarray  # (E,)
    target_angles: np.ndarray  # (A,) radians
    torsion_quads: List[Tuple[int, int, int, int]] = field(default_factory=list)
    torsion_params: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64)
    )  # MMFF V1/V2/V3
    chiral_centers: List[Tuple[int, int, int, int]] = field(default_factory=list)
    chiral_signs: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    planar_groups: List[Tuple[int, ...]] = field(default_factory=list)
    nonbonded_pairs: List[Tuple[int, int]] = field(default_factory=list)
    clash_distances: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))


@dataclass
class SolveResult:
    """Output of solve_3d."""

    coords_3d: np.ndarray  # (N, 3)
    final_loss: float
    loss_reproj: float
    loss_bond: float
    loss_angle: float
    best_start_idx: int
    loss_torsion: float = 0.0
    loss_chirality: float = 0.0
    loss_planarity: float = 0.0
    loss_clash: float = 0.0
    loss_history: List[float] = field(default_factory=list)
    frame: Optional[Plane2DFrame] = None
    mol: Any = None
    smiles: str = ""
    # Full multi-start population for downstream refinement and reranking.
    # coords_3d and final_loss refer to the Solver-selected candidate.
    candidate_coords_3d: Optional[np.ndarray] = None  # (S, N, 3)
    candidate_final_losses: Optional[np.ndarray] = None  # (S,)
    diagnostics: dict = field(default_factory=dict)
