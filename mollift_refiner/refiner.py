"""Short-range Open Babel refinement with strict observation-preserving gates.

The refiner is deliberately a local geometry repair stage, not a conformer
search.  Heavy atoms are first fixed while hydrogens relax.  Heavy atoms are
then released for a small number of steepest-descent steps, and every step is
accepted only while displacement, projection, and clash gates remain valid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any, Optional, Tuple

import numpy as np
from rdkit import Chem

from mollift_solver.pipeline import MoleculeInput, SolverOutput
from mollift_solver.projection import Plane2DFrame, project_3d_to_2d_numpy
from mollift_solver.topology import build_topology


@dataclass(frozen=True)
class RefinerResult:
    """Public result in the authoritative heavy-only or explicit-H atom order."""

    molecule: MoleculeInput
    coords_3d: np.ndarray
    energy: Optional[float] = None
    accepted: bool = True
    status: str = "success"
    failure_reason: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OpenBabelRefinerConfig:
    """Configuration for conservative local force-field relaxation."""

    primary_force_field: str = "MMFF94s"
    fallback_force_field: Optional[str] = "UFF"
    hydrogen_steps: int = 50
    heavy_steps: int = 5
    convergence: float = 1e-4
    max_heavy_rms_displacement: float = 0.05
    max_atom_displacement: float = 0.15
    max_projection_rmse_increase: float = 0.03
    max_clash_increase: float = 1e-6
    require_energy_nonincrease: bool = True
    max_backtracking_steps: int = 8
    backtracking_factor: float = 0.5
    refine_top_m: int = 3
    raise_on_failure: bool = False

    def validate(self) -> None:
        if not self.primary_force_field:
            raise ValueError("primary_force_field must be non-empty")
        if self.hydrogen_steps < 0 or self.heavy_steps < 0:
            raise ValueError("refinement step counts must be non-negative")
        if self.max_backtracking_steps < 0:
            raise ValueError("max_backtracking_steps must be non-negative")
        if not 0.0 < self.backtracking_factor < 1.0:
            raise ValueError("backtracking_factor must be between zero and one")
        if self.refine_top_m < 1:
            raise ValueError("refine_top_m must be at least 1")
        if self.convergence <= 0:
            raise ValueError("convergence must be positive")
        for name in (
            "max_heavy_rms_displacement",
            "max_atom_displacement",
            "max_projection_rmse_increase",
            "max_clash_increase",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


def openbabel_version() -> str:
    """Return the loaded Open Babel runtime version without importing it eagerly."""
    from openbabel import openbabel as ob

    return str(ob.OBReleaseVersion())


def _rdkit_mol_with_coordinates(molecule: MoleculeInput, coords: np.ndarray) -> Chem.Mol:
    """Attach coordinates to the authoritative atom set, adding H only if implicit."""
    mol = molecule.to_rdkit_mol()
    if mol.GetNumAtoms() != len(coords):
        raise ValueError("Coordinate count does not match the resolved molecule atom count")
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.Set3D(True)
    for idx, xyz in enumerate(np.asarray(coords, dtype=np.float64)):
        conf.SetAtomPosition(idx, tuple(float(v) for v in xyz))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    if any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
        return mol
    # Append force-field hydrogens after the supplied heavy atoms.
    return Chem.AddHs(mol, addCoords=True)


def _to_openbabel(rdkit_mol: Chem.Mol, expected_atomic_numbers: Tuple[int, ...]):
    from openbabel import openbabel as ob

    conversion = ob.OBConversion()
    if not conversion.SetInFormat("mol"):
        raise RuntimeError("Open Babel mol reader is unavailable")
    obmol = ob.OBMol()
    if not conversion.ReadString(obmol, Chem.MolToMolBlock(rdkit_mol, includeStereo=True)):
        raise RuntimeError("Open Babel failed to read the RDKit mol block")
    observed = tuple(obmol.GetAtom(i + 1).GetAtomicNum() for i in range(len(expected_atomic_numbers)))
    if observed != expected_atomic_numbers:
        raise RuntimeError(
            "Open Babel changed the authoritative atom prefix/order: "
            f"expected {expected_atomic_numbers}, got {observed}"
        )
    if any(obmol.GetAtom(i + 1).GetAtomicNum() != 1 for i in range(len(expected_atomic_numbers), obmol.NumAtoms())):
        raise RuntimeError("Open Babel did not preserve the heavy-atom then hydrogen ordering")
    return obmol


def _prefix_coords(obmol, count: int) -> np.ndarray:
    return np.asarray(
        [
            [obmol.GetAtom(i + 1).GetX(), obmol.GetAtom(i + 1).GetY(), obmol.GetAtom(i + 1).GetZ()]
            for i in range(count)
        ],
        dtype=np.float64,
    )


def _projection_rmse(
    coords: np.ndarray,
    target_2d: np.ndarray,
    mask: np.ndarray,
    frame: Plane2DFrame,
) -> float:
    if not bool(mask.any()):
        return 0.0
    projected = project_3d_to_2d_numpy(coords, frame)
    diff = projected[mask] - target_2d[mask]
    return float(np.sqrt(np.mean(diff * diff)))


def _clash_metrics(topology, coords: np.ndarray) -> Tuple[int, float]:
    if not topology.nonbonded_pairs:
        return 0, 0.0
    pairs = np.asarray(topology.nonbonded_pairs, dtype=np.int64)
    distances = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)
    penetration = np.maximum(np.asarray(topology.clash_distances) - distances, 0.0)
    return int(np.count_nonzero(penetration > 1e-8)), float(penetration.max(initial=0.0))


def _chirality_violations(topology, coords: np.ndarray) -> int:
    """Count specified tetrahedral centers whose oriented volume has wrong sign."""
    violations = 0
    for (a, center, b, c), expected_sign in zip(
        topology.chiral_centers, topology.chiral_signs
    ):
        va = coords[a] - coords[center]
        vb = coords[b] - coords[center]
        vc = coords[c] - coords[center]
        scale = np.linalg.norm(va) * np.linalg.norm(vb) * np.linalg.norm(vc)
        signed = float(np.dot(np.cross(va, vb), vc) / max(scale, 1e-12))
        if float(expected_sign) * signed <= 0.0:
            violations += 1
    return violations


def _specified_ez_bonds(smiles: str) -> list[tuple[int, int, int, int, float]]:
    """Return SMILES-specified double-bond stereo as geometric sign constraints.

    Each tuple is ``(left_substituent, left, right, right_substituent,
    expected_dot_sign)``. After removing the component parallel to the double-bond
    axis, Z/cis substituent vectors point to the same side (positive dot product),
    while E/trans vectors point to opposite sides (negative dot product).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    same_side = {Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS}
    opposite_side = {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS}
    constraints: list[tuple[int, int, int, int, float]] = []
    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        if stereo not in same_side | opposite_side:
            continue
        stereo_atoms = tuple(int(index) for index in bond.GetStereoAtoms())
        if len(stereo_atoms) != 2:
            continue
        constraints.append(
            (
                stereo_atoms[0],
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                stereo_atoms[1],
                1.0 if stereo in same_side else -1.0,
            )
        )
    return constraints


def _ez_violations(
    constraints: list[tuple[int, int, int, int, float]], coords: np.ndarray
) -> int:
    """Count specified E/Z bonds whose current 3D substituent sides disagree."""
    violations = 0
    for left_sub, left, right, right_sub, expected_sign in constraints:
        axis = coords[right] - coords[left]
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-12:
            violations += 1
            continue
        axis = axis / axis_norm
        left_vector = coords[left_sub] - coords[left]
        right_vector = coords[right_sub] - coords[right]
        left_perp = left_vector - float(np.dot(left_vector, axis)) * axis
        right_perp = right_vector - float(np.dot(right_vector, axis)) * axis
        scale = float(np.linalg.norm(left_perp) * np.linalg.norm(right_perp))
        if scale <= 1e-12:
            violations += 1
            continue
        side_cosine = float(np.dot(left_perp, right_perp) / scale)
        if expected_sign * side_cosine <= 0.0:
            violations += 1
    return violations


def _setup_force_field(obmol, config: OpenBabelRefinerConfig, fixed_atom_indices: list[int]):
    from openbabel import openbabel as ob

    candidates = [config.primary_force_field]
    if config.fallback_force_field and config.fallback_force_field not in candidates:
        candidates.append(config.fallback_force_field)
    errors = []
    for name in candidates:
        force_field = ob.OBForceField.FindForceField(name)
        if force_field is None:
            errors.append(f"{name}: unavailable")
            continue
        constraints = ob.OBFFConstraints()
        for atom_idx in fixed_atom_indices:
            constraints.AddAtomConstraint(atom_idx + 1)
        if force_field.Setup(obmol, constraints):
            return force_field, str(force_field.GetID()), str(force_field.GetUnit())
        errors.append(f"{name}: parameterization/setup failed")
    raise RuntimeError("; ".join(errors))


class OpenBabelRefiner:
    """Projection-preserving, short-range Open Babel force-field refiner."""

    def __init__(self, config: Optional[OpenBabelRefinerConfig] = None):
        self.config = config or OpenBabelRefinerConfig()
        self.config.validate()

    def _failure(self, solved: SolverOutput, reason: str, started_at: float) -> RefinerResult:
        if self.config.raise_on_failure:
            raise RuntimeError(reason)
        return RefinerResult(
            molecule=solved.molecule,
            coords_3d=np.asarray(solved.coords_3d, dtype=np.float64).copy(),
            energy=None,
            accepted=False,
            status="failed",
            failure_reason=reason,
            metadata={
                "backend": "openbabel",
                "config": asdict(self.config),
                "runtime_seconds": perf_counter() - started_at,
            },
        )

    def _refine_one(self, solver_output: SolverOutput) -> RefinerResult:
        started_at = perf_counter()
        try:
            from openbabel import openbabel as ob

            self.config.validate()
            molecule = solver_output.molecule
            molecule.validate()
            initial = np.asarray(solver_output.coords_3d, dtype=np.float64)
            n_input = len(molecule.atomic_numbers)
            if initial.shape != (n_input, 3) or not np.isfinite(initial).all():
                raise ValueError(f"Solver coordinates must be finite with shape ({n_input}, 3)")

            rdkit_h = _rdkit_mol_with_coordinates(molecule, initial)
            expected = tuple(int(z) for z in molecule.atomic_numbers)
            obmol = _to_openbabel(rdkit_h, expected)
            heavy_indices = [i for i, z in enumerate(expected) if z != 1]
            force_field, force_field_name, energy_unit = _setup_force_field(
                obmol, self.config, heavy_indices
            )
            energy_before = float(force_field.Energy())

            # R0: optimize only appended hydrogens while every heavy atom is fixed.
            hydrogen_converged = True
            if self.config.hydrogen_steps:
                force_field.SteepestDescentInitialize(
                    self.config.hydrogen_steps, self.config.convergence
                )
                hydrogen_converged = not bool(
                    force_field.SteepestDescentTakeNSteps(self.config.hydrogen_steps)
                )
                force_field.GetCoordinates(obmol)
            energy_after_hydrogen = float(force_field.Energy())

            # Clear R0 constraints explicitly; Open Babel retains them across Setup calls.
            empty_constraints = ob.OBFFConstraints()
            if not force_field.Setup(obmol, empty_constraints):
                raise RuntimeError(f"{force_field_name}: unconstrained setup failed")

            frame_value = solver_output.diagnostics.get("frame")
            frame = frame_value if isinstance(frame_value, Plane2DFrame) else Plane2DFrame.default_xy_plane()
            target_2d = np.asarray(molecule.coords_2d, dtype=np.float64)
            if molecule.coords_2d_mask is None:
                projection_mask = np.isfinite(target_2d).all(axis=1)
            else:
                projection_mask = np.asarray(molecule.coords_2d_mask, dtype=bool)
            projection_before = _projection_rmse(initial, target_2d, projection_mask, frame)
            topology = build_topology(molecule.smiles, mol=molecule.to_rdkit_mol())
            ez_constraints = _specified_ez_bonds(molecule.smiles)
            clash_count_before, clash_before = _clash_metrics(topology, initial)
            chirality_before = _chirality_violations(topology, initial)
            ez_before = _ez_violations(ez_constraints, initial)

            last_coords = initial.copy()
            last_energy = energy_after_hydrogen
            accepted_steps = 0
            backtracking_trials = 0
            accepted_step_scales: list[float] = []
            rejected_gate: Optional[str] = None
            heavy_converged = self.config.heavy_steps == 0
            if self.config.heavy_steps:
                for _ in range(self.config.heavy_steps):
                    # Backtrack each proposal to enforce the Cartesian trust region.
                    force_field.SteepestDescentInitialize(1, self.config.convergence)
                    force_field.SteepestDescentTakeNSteps(1)
                    force_field.GetCoordinates(obmol)
                    proposal = _prefix_coords(obmol, n_input)
                    if not np.isfinite(proposal).all():
                        rejected_gate = "non_finite_coordinates"
                        break
                    proposal_step_norm = float(
                        np.max(np.linalg.norm(proposal - last_coords, axis=1))
                    )
                    accepted_candidate = None
                    last_failed_gate = None
                    accepted_energy = None
                    accepted_scale = None
                    for backtrack_index in range(self.config.max_backtracking_steps + 1):
                        scale = self.config.backtracking_factor**backtrack_index
                        candidate = last_coords + scale * (proposal - last_coords)
                        for atom_index, xyz in enumerate(candidate, start=1):
                            obmol.GetAtom(atom_index).SetVector(*(float(value) for value in xyz))
                        if not force_field.Setup(obmol, empty_constraints):
                            last_failed_gate = "force_field_setup"
                            continue
                        candidate_energy = float(force_field.Energy())
                        displacement = np.linalg.norm(candidate - initial, axis=1)
                        heavy_displacement = displacement[heavy_indices]
                        rms_displacement = float(np.sqrt(np.mean(heavy_displacement * heavy_displacement)))
                        max_displacement = float(displacement.max(initial=0.0))
                        projection_after = _projection_rmse(
                            candidate, target_2d, projection_mask, frame
                        )
                        _, clash_after = _clash_metrics(topology, candidate)
                        chirality_after = _chirality_violations(topology, candidate)
                        ez_after = _ez_violations(ez_constraints, candidate)
                        checks = (
                            (rms_displacement <= self.config.max_heavy_rms_displacement, "rms_displacement"),
                            (max_displacement <= self.config.max_atom_displacement, "max_atom_displacement"),
                            (
                                projection_after - projection_before
                                <= self.config.max_projection_rmse_increase,
                                "projection_rmse",
                            ),
                            (
                                clash_after - clash_before <= self.config.max_clash_increase,
                                "clash",
                            ),
                            (chirality_after <= chirality_before, "chirality"),
                            (ez_after <= ez_before, "ez_stereo"),
                            (
                                not self.config.require_energy_nonincrease
                                or candidate_energy <= last_energy + 1e-8,
                                "energy",
                            ),
                        )
                        last_failed_gate = next(
                            (name for ok, name in checks if not ok), None
                        )
                        if last_failed_gate is None:
                            accepted_candidate = candidate
                            accepted_energy = candidate_energy
                            accepted_scale = scale
                            backtracking_trials += backtrack_index
                            break
                    if accepted_candidate is None:
                        rejected_gate = last_failed_gate
                        break
                    last_coords = accepted_candidate
                    last_energy = float(accepted_energy)
                    accepted_steps += 1
                    accepted_step_scales.append(float(accepted_scale))
                    if proposal_step_norm < 1e-10:
                        heavy_converged = True
                        break

            displacement = np.linalg.norm(last_coords - initial, axis=1)
            projection_after = _projection_rmse(last_coords, target_2d, projection_mask, frame)
            clash_count_after, clash_after = _clash_metrics(topology, last_coords)
            chirality_after = _chirality_violations(topology, last_coords)
            ez_after = _ez_violations(ez_constraints, last_coords)
            status = "refined" if accepted_steps else "hydrogen_only"
            return RefinerResult(
                molecule=molecule,
                coords_3d=last_coords,
                energy=last_energy,
                accepted=True,
                status=status,
                failure_reason=None,
                metadata={
                    "backend": "openbabel",
                    "openbabel_version": str(ob.OBReleaseVersion()),
                    "force_field": force_field_name,
                    "energy_unit": energy_unit,
                    "energy_before": energy_before,
                    "energy_after_hydrogen": energy_after_hydrogen,
                    "energy_after": last_energy,
                    "hydrogen_converged": hydrogen_converged,
                    "heavy_converged": heavy_converged,
                    "heavy_steps_accepted": accepted_steps,
                    "backtracking_trials": backtracking_trials,
                    "accepted_step_scales": accepted_step_scales,
                    "next_step_rejected_by": rejected_gate,
                    "heavy_rms_displacement": float(np.sqrt(np.mean(displacement[heavy_indices] * displacement[heavy_indices]))),
                    "max_atom_displacement": float(displacement.max(initial=0.0)),
                    "projection_rmse_before": projection_before,
                    "projection_rmse_after": projection_after,
                    "clash_count_before": clash_count_before,
                    "clash_count_after": clash_count_after,
                    "max_clash_penetration_before": clash_before,
                    "max_clash_penetration_after": clash_after,
                    "chirality_violations_before": chirality_before,
                    "chirality_violations_after": chirality_after,
                    "specified_ez_bonds": len(ez_constraints),
                    "ez_violations_before": ez_before,
                    "ez_violations_after": ez_after,
                    "config": asdict(self.config),
                    "runtime_seconds": perf_counter() - started_at,
                },
            )
        except Exception as exc:
            return self._failure(solver_output, f"{type(exc).__name__}: {exc}", started_at)

    def refine(self, solver_output: SolverOutput) -> RefinerResult:
        """Refine the public Solver winner."""
        return self._refine_one(solver_output)

    def refine_all_candidates(
        self,
        molecule: MoleculeInput,
        candidate_coords_3d: np.ndarray,
        candidate_objectives: Optional[np.ndarray] = None,
        diagnostics: Optional[dict[str, Any]] = None,
    ) -> list[RefinerResult]:
        """Refine every finite candidate while preserving the candidate axis.

        No Solver-objective top-M prefilter is applied. Invalid candidates
        receive explicit failed results at their original positions so later
        selection and fixed-denominator accounting cannot silently shrink K.
        """
        candidates = np.asarray(candidate_coords_3d, dtype=np.float64)
        n_heavy = len(molecule.atomic_numbers)
        if candidates.ndim != 3 or candidates.shape[1:] != (n_heavy, 3):
            raise ValueError(
                f"candidate_coords_3d must have shape (K,{n_heavy},3), got {candidates.shape}"
            )
        if candidate_objectives is None:
            objectives = np.zeros(len(candidates), dtype=np.float64)
        else:
            objectives = np.asarray(candidate_objectives, dtype=np.float64)
            if objectives.shape != (len(candidates),):
                raise ValueError("candidate_objectives must have shape (K,)")

        outputs: list[RefinerResult] = []
        for candidate_idx, coords in enumerate(candidates):
            candidate_diagnostics = dict(diagnostics or {})
            candidate_diagnostics["refiner_solver_candidate_idx"] = candidate_idx
            solved = SolverOutput(
                molecule=molecule,
                coords_3d=coords,
                objective=float(objectives[candidate_idx]),
                diagnostics=candidate_diagnostics,
            )
            if not np.isfinite(coords).all() or not np.isfinite(objectives[candidate_idx]):
                result = self._failure(solved, "non-finite input candidate", perf_counter())
            else:
                result = self._refine_one(solved)
            result.metadata["source_candidate_idx"] = candidate_idx
            outputs.append(result)
        return outputs

    def refine_candidates(
        self,
        molecule: MoleculeInput,
        candidate_coords_3d: np.ndarray,
        candidate_objectives: np.ndarray,
        diagnostics: Optional[dict[str, Any]] = None,
    ) -> RefinerResult:
        """Refine top-M candidates without adding a candidate axis to Solver types."""
        candidates = candidate_coords_3d
        objectives = candidate_objectives
        candidates = np.asarray(candidates, dtype=np.float64)
        objectives = np.asarray(objectives, dtype=np.float64)
        n_heavy = len(molecule.atomic_numbers)
        if (
            candidates.ndim != 3
            or candidates.shape[1:] != (n_heavy, 3)
            or objectives.shape != (candidates.shape[0],)
        ):
            placeholder = SolverOutput(
                molecule, np.zeros((n_heavy, 3)), float("inf"), diagnostics or {}
            )
            return self._failure(
                placeholder,
                "Invalid candidate_coords_3d/candidate_objectives shapes",
                perf_counter(),
            )

        finite = np.isfinite(objectives) & np.isfinite(candidates).all(axis=(1, 2))
        ranked = np.flatnonzero(finite)
        ranked = ranked[np.argsort(objectives[ranked], kind="stable")]
        ranked = ranked[: self.config.refine_top_m]
        if len(ranked) == 0:
            placeholder = SolverOutput(molecule, np.zeros((n_heavy, 3)), float("inf"), diagnostics or {})
            return self._failure(placeholder, "No finite Solver candidate to refine", perf_counter())

        outputs = []
        for candidate_idx in ranked.tolist():
            candidate_diagnostics = dict(diagnostics or {})
            candidate_diagnostics["refiner_solver_candidate_idx"] = int(candidate_idx)
            candidate_output = SolverOutput(
                molecule=molecule,
                coords_3d=candidates[candidate_idx],
                objective=float(objectives[candidate_idx]),
                diagnostics=candidate_diagnostics,
            )
            outputs.append((candidate_idx, self._refine_one(candidate_output)))

        accepted = [(idx, output) for idx, output in outputs if output.accepted]
        if not accepted:
            first_reason = outputs[0][1].failure_reason or "all candidates rejected"
            result = self._failure(
                SolverOutput(
                    molecule, candidates[ranked[0]], float(objectives[ranked[0]]), diagnostics or {}
                ),
                f"All top-{len(outputs)} candidates failed: {first_reason}",
                perf_counter(),
            )
            summaries = [
                {
                    "solver_candidate_idx": idx,
                    "solver_objective": float(objectives[idx]),
                    "accepted": output.accepted,
                    "status": output.status,
                    "failure_reason": output.failure_reason,
                }
                for idx, output in outputs
            ]
            return RefinerResult(
                molecule=result.molecule,
                coords_3d=result.coords_3d,
                energy=result.energy,
                accepted=result.accepted,
                status=result.status,
                failure_reason=result.failure_reason,
                metadata={**result.metadata, "candidate_summaries": summaries},
            )

        # A molecule should parameterize consistently across candidates. Refuse
        # cross-force-field energy comparisons if an unexpected mismatch occurs.
        force_fields = {str(output.metadata.get("force_field")) for _, output in accepted}
        units = {str(output.metadata.get("energy_unit")) for _, output in accepted}
        if len(force_fields) != 1 or len(units) != 1:
            return self._failure(
                SolverOutput(molecule, candidates[ranked[0]], float(objectives[ranked[0]]), diagnostics or {}),
                "Accepted candidates used different force fields or energy units",
                perf_counter(),
            )
        selected_idx, selected = min(
            accepted,
            key=lambda item: float(item[1].energy) if item[1].energy is not None else float("inf"),
        )
        summaries = [
            {
                "solver_candidate_idx": idx,
                "solver_objective": float(objectives[idx]),
                "accepted": output.accepted,
                "status": output.status,
                "failure_reason": output.failure_reason,
                "energy": output.energy,
                "force_field": output.metadata.get("force_field"),
                "energy_unit": output.metadata.get("energy_unit"),
                "heavy_rms_displacement": output.metadata.get("heavy_rms_displacement"),
                "projection_rmse_after": output.metadata.get("projection_rmse_after"),
            }
            for idx, output in outputs
        ]
        return RefinerResult(
            molecule=selected.molecule,
            coords_3d=selected.coords_3d,
            energy=selected.energy,
            accepted=True,
            status=selected.status,
            failure_reason=None,
            metadata={
                **selected.metadata,
                "selection": "lowest_energy_among_gate_passed",
                "selected_solver_candidate_idx": int(selected_idx),
                "solver_candidates_available": int(len(candidates)),
                "solver_candidates_refined": int(len(outputs)),
                "candidate_summaries": summaries,
            },
        )
