"""Atom-ordered inputs and outputs for Generator, Solver, and Refiner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

import numpy as np
from rdkit import Chem

from mollift_solver.solver import solve_3d
from mollift_solver.types import SolveConfig, SolveResult


@dataclass(frozen=True)
class MoleculeInput:
    """A molecule and projection in authoritative explicit-atom row order."""

    smiles: str
    coords_2d: np.ndarray
    atomic_numbers: Sequence[int]
    atom_ids: Optional[Sequence[str]] = None
    coords_2d_mask: Optional[np.ndarray] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_rdkit_mol(self) -> Chem.Mol:
        """Resolve heavy-only or explicit-H topology without changing atom order."""
        heavy = Chem.MolFromSmiles(self.smiles)
        if heavy is None:
            raise ValueError(f"Invalid SMILES: {self.smiles!r}")
        supplied = tuple(int(z) for z in self.atomic_numbers)
        heavy_numbers = tuple(atom.GetAtomicNum() for atom in heavy.GetAtoms())
        if supplied == heavy_numbers:
            return heavy
        full = Chem.AddHs(Chem.Mol(heavy))
        full_numbers = tuple(atom.GetAtomicNum() for atom in full.GetAtoms())
        if supplied == full_numbers:
            return full
        raise ValueError(
            "atomic_numbers must match either RDKit MolFromSmiles or its AddHs order; "
            f"heavy={heavy_numbers}, explicit-H={full_numbers}, got={supplied}"
        )

    def validate(self) -> None:
        mol = self.to_rdkit_mol()
        expected = tuple(atom.GetAtomicNum() for atom in mol.GetAtoms())
        supplied = tuple(int(z) for z in self.atomic_numbers)
        if supplied != expected:
            raise ValueError(
                "atomic_numbers must exactly match the resolved RDKit atom order; "
                f"expected {expected}, got {supplied}"
            )
        n = len(expected)
        if np.asarray(self.coords_2d).shape != (n, 2):
            raise ValueError(f"coords_2d must have shape ({n}, 2)")
        if self.atom_ids is not None and len(self.atom_ids) != n:
            raise ValueError(f"atom_ids must contain {n} entries")


@dataclass(frozen=True)
class GeneratorOutput:
    """Generator candidates, shape ``(K, N, 3)`` in MoleculeInput order."""

    molecule: MoleculeInput
    coords_3d: np.ndarray
    scores: Optional[np.ndarray] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.molecule.validate()
        coords = np.asarray(self.coords_3d)
        n = len(self.molecule.atomic_numbers)
        if coords.ndim != 3 or coords.shape[1:] != (n, 3) or coords.shape[0] == 0:
            raise ValueError(f"coords_3d must have shape (K, {n}, 3) with K >= 1")
        if self.scores is not None and np.asarray(self.scores).shape != (coords.shape[0],):
            raise ValueError("scores must have shape (K,)")


@dataclass(frozen=True)
class SolverOutput:
    molecule: MoleculeInput
    coords_3d: np.ndarray
    objective: float
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class RefinerOutput:
    molecule: MoleculeInput
    coords_3d: np.ndarray
    energy: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Refiner(Protocol):
    def refine(self, solver_output: SolverOutput) -> RefinerOutput: ...


def run_solver_stage(generated: GeneratorOutput, config: Optional[SolveConfig] = None) -> SolverOutput:
    """Optimize Generator candidates with the batched multi-start Solver."""
    generated.validate()
    molecule = generated.molecule
    result: SolveResult = solve_3d(
        molecule.smiles,
        molecule.coords_2d,
        coords_3d_init=[x for x in np.asarray(generated.coords_3d)],
        config=config,
        coords_2d_mask=molecule.coords_2d_mask,
        mol=molecule.to_rdkit_mol(),
    )
    return SolverOutput(
        molecule=molecule,
        coords_3d=result.coords_3d,
        objective=result.final_loss,
        diagnostics={
            "best_start_idx": result.best_start_idx,
            "loss_reproj": result.loss_reproj,
            "loss_bond": result.loss_bond,
            "loss_angle": result.loss_angle,
            "loss_torsion": result.loss_torsion,
            "loss_chirality": result.loss_chirality,
            "loss_planarity": result.loss_planarity,
            "loss_clash": result.loss_clash,
        },
    )


def run_pipeline(
    generated: GeneratorOutput,
    solver_config: Optional[SolveConfig] = None,
    refiner: Optional[Refiner] = None,
) -> SolverOutput | RefinerOutput:
    """Run Solver followed by optional refinement of its selected candidate."""
    solved = run_solver_stage(generated, solver_config)
    return solved if refiner is None else refiner.refine(solved)
