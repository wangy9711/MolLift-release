"""Public entry point: solve_3d."""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
from rdkit import Chem

from mollift_solver.initialize import _resolve_reproj_mask, generate_initial_coords
from mollift_solver.optimize import optimize_multi_start
from mollift_solver.projection import Plane2DFrame
from mollift_solver.topology import build_topology, validate_coords_2d, validate_coords_2d_partial
from mollift_solver.types import SolveConfig, SolveResult


def solve_3d(
    smiles: str,
    coords_2d: np.ndarray,
    coords_3d_init: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    config: Optional[SolveConfig] = None,
    coords_2d_mask: Optional[np.ndarray] = None,
    mol: Optional[Chem.Mol] = None,
) -> SolveResult:
    """
    Recover 3D coordinates from SMILES topology + observed 2D positions.

    Parameters
    ----------
    smiles : str
        SMILES string; atom order follows RDKit MolFromSmiles.
    coords_2d : (N, 2) array
        Observed 2D coordinates; row i matches atom index i.
        Unobserved rows may be NaN when using partial observations.
    coords_3d_init : optional (N, 3) or list of (N, 3)
        Initial 3D guess(es), e.g. from DL; best after optimization is returned.
    config : SolveConfig, optional
        Optimization hyperparameters.
    coords_2d_mask : optional (N,) bool
        True for atoms with valid 2D observations. If None, inferred from NaN
        rows or defaults to all atoms observed.

    Returns
    -------
    SolveResult with coords_3d (N, 3) in Angstrom-like units (same scale as 2D input).
    """
    if config is None:
        config = SolveConfig()

    topology = build_topology(smiles, mol=mol)
    coords_2d = np.asarray(coords_2d, dtype=np.float64)
    mask = _resolve_reproj_mask(coords_2d, coords_2d_mask)

    if mask.all():
        validate_coords_2d(topology.mol, coords_2d)
    else:
        validate_coords_2d_partial(topology.mol, coords_2d, mask)

    frame = config.plane_frame or Plane2DFrame.default_xy_plane()

    inits = generate_initial_coords(
        topology.mol,
        coords_2d,
        config,
        frame,
        coords_3d_init=coords_3d_init,
        coords_2d_mask=mask,
    )

    return optimize_multi_start(
        inits, coords_2d, topology, frame, config, reproj_mask=mask
    )
