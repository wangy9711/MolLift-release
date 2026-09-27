import numpy as np
import pytest
pytest.importorskip("openbabel")
from rdkit import Chem
from rdkit.Chem import AllChem

from mollift_refiner import OpenBabelRefiner, OpenBabelRefinerConfig, openbabel_version
from mollift_refiner.refiner import _ez_violations, _specified_ez_bonds
from mollift_solver import MoleculeInput, SolverOutput
from mollift_solver.projection import Plane2DFrame, project_3d_to_2d_numpy
from mollift_solver.topology import get_coords_from_mol


def _solved(smiles: str = "CCCO") -> SolverOutput:
    mol = Chem.MolFromSmiles(smiles)
    params = AllChem.ETKDGv3()
    params.randomSeed = 23
    assert AllChem.EmbedMolecule(mol, params) == 0
    coords = get_coords_from_mol(mol)
    frame = Plane2DFrame.default_xy_plane()
    projection = project_3d_to_2d_numpy(coords, frame)
    molecule = MoleculeInput(
        smiles=smiles,
        coords_2d=projection,
        atomic_numbers=[atom.GetAtomicNum() for atom in mol.GetAtoms()],
    )
    return SolverOutput(molecule, coords, 0.0, {"frame": frame})


def test_openbabel_runtime_and_force_field_are_available():
    assert openbabel_version().startswith("3.1")
    result = OpenBabelRefiner(OpenBabelRefinerConfig(heavy_steps=0)).refine(_solved())
    assert result.accepted
    assert result.metadata["force_field"] == "MMFF94s"
    assert result.metadata["energy_unit"] == "kcal/mol"


def test_refiner_preserves_order_projection_and_displacement_gates():
    solved = _solved("CCOC(=O)N")
    config = OpenBabelRefinerConfig(
        hydrogen_steps=10,
        heavy_steps=25,
        max_heavy_rms_displacement=0.08,
        max_atom_displacement=0.15,
        max_projection_rmse_increase=0.02,
    )
    result = OpenBabelRefiner(config).refine(solved)
    assert result.accepted
    assert result.coords_3d.shape == solved.coords_3d.shape
    assert result.molecule is solved.molecule
    assert result.metadata["heavy_rms_displacement"] <= 0.08 + 1e-12
    assert result.metadata["max_atom_displacement"] <= 0.15 + 1e-12
    assert (
        result.metadata["projection_rmse_after"]
        - result.metadata["projection_rmse_before"]
        <= 0.02 + 1e-12
    )
    assert result.metadata["max_clash_penetration_after"] <= (
        result.metadata["max_clash_penetration_before"] + config.max_clash_increase + 1e-12
    )


def test_force_field_fallback_is_explicit_and_keeps_its_energy_unit():
    config = OpenBabelRefinerConfig(
        primary_force_field="not-a-force-field",
        fallback_force_field="UFF",
        hydrogen_steps=2,
        heavy_steps=0,
    )
    result = OpenBabelRefiner(config).refine(_solved())
    assert result.accepted
    assert result.metadata["force_field"] == "UFF"
    assert result.metadata["energy_unit"] == "kJ/mol"


def test_backend_failure_is_explicit_and_returns_original_coordinates():
    solved = _solved()
    config = OpenBabelRefinerConfig(
        primary_force_field="missing-primary",
        fallback_force_field="missing-fallback",
    )
    result = OpenBabelRefiner(config).refine(solved)
    assert not result.accepted
    assert result.status == "failed"
    assert "unavailable" in result.failure_reason
    assert np.array_equal(result.coords_3d, solved.coords_3d)


def test_zero_displacement_budget_never_exposes_a_moved_heavy_structure():
    solved = _solved("CCN(CC)CC")
    config = OpenBabelRefinerConfig(
        hydrogen_steps=5,
        heavy_steps=10,
        max_heavy_rms_displacement=0.0,
        max_atom_displacement=0.0,
        max_projection_rmse_increase=0.0,
    )
    result = OpenBabelRefiner(config).refine(solved)
    assert result.accepted
    assert result.status == "hydrogen_only"
    assert result.metadata["heavy_steps_accepted"] == 0
    assert np.array_equal(result.coords_3d, solved.coords_3d)


def test_large_force_field_step_is_backtracked_into_local_trust_region():
    solved = _solved("CCN(CC)CC")
    config = OpenBabelRefinerConfig(
        hydrogen_steps=5,
        heavy_steps=1,
        max_heavy_rms_displacement=0.005,
        max_atom_displacement=0.01,
        max_projection_rmse_increase=1.0,
        max_clash_increase=1.0,
        max_backtracking_steps=12,
    )
    result = OpenBabelRefiner(config).refine(solved)
    assert result.accepted
    assert result.metadata["heavy_steps_accepted"] == 1
    assert result.metadata["backtracking_trials"] > 0
    assert result.metadata["accepted_step_scales"][0] < 1.0
    assert result.metadata["heavy_rms_displacement"] <= 0.005 + 1e-12
    assert result.metadata["max_atom_displacement"] <= 0.01 + 1e-12


def test_top_m_candidates_are_refined_and_selected_by_comparable_energy():
    solved = _solved("CCCCO")
    candidates = np.stack(
        [
            solved.coords_3d,
            solved.coords_3d + np.array([0.0, 0.0, 0.03]),
            solved.coords_3d + np.array([0.02, 0.0, 0.0]),
            solved.coords_3d + np.array([0.0, 0.02, 0.0]),
        ]
    )
    result = OpenBabelRefiner(
        OpenBabelRefinerConfig(hydrogen_steps=2, heavy_steps=2, refine_top_m=3)
    ).refine_candidates(
        solved.molecule,
        candidates,
        np.array([0.3, 0.1, 0.2, 0.4]),
        diagnostics=dict(solved.diagnostics),
    )
    assert result.accepted
    assert result.metadata["solver_candidates_available"] == 4
    assert result.metadata["solver_candidates_refined"] == 3
    assert result.metadata["selected_solver_candidate_idx"] in {0, 1, 2}
    assert result.metadata["selection"] == "lowest_energy_among_gate_passed"
    energies = [x["energy"] for x in result.metadata["candidate_summaries"] if x["accepted"]]
    assert np.isclose(result.energy, min(energies))


def test_all_candidates_are_refined_without_top_m_prefilter():
    solved = _solved("CCCO")
    candidates = np.stack(
        [solved.coords_3d + np.array([0.0, 0.0, 0.01 * i]) for i in range(4)]
    )
    outputs = OpenBabelRefiner(
        OpenBabelRefinerConfig(hydrogen_steps=1, heavy_steps=0, refine_top_m=1)
    ).refine_all_candidates(
        solved.molecule,
        candidates,
        np.array([0.4, 0.3, 0.2, 0.1]),
        diagnostics=dict(solved.diagnostics),
    )
    assert len(outputs) == 4
    assert [row.metadata["source_candidate_idx"] for row in outputs] == list(range(4))
    assert all(row.accepted for row in outputs)


def test_specified_chirality_cannot_be_made_worse():
    solved = _solved("C[C@H](O)F")
    result = OpenBabelRefiner(
        OpenBabelRefinerConfig(hydrogen_steps=5, heavy_steps=10)
    ).refine(solved)
    assert result.accepted
    assert result.metadata["chirality_violations_after"] <= result.metadata[
        "chirality_violations_before"
    ]


def test_specified_ez_geometry_is_detected_and_cannot_be_made_worse():
    trans = np.asarray(
        [[-1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, -1.0, 0.0]]
    )
    cis = trans.copy()
    cis[3, 1] = 1.0
    constraints = _specified_ez_bonds("F/C=C/F")
    assert len(constraints) == 1
    assert _ez_violations(constraints, trans) == 0
    assert _ez_violations(constraints, cis) == 1

    result = OpenBabelRefiner(
        OpenBabelRefinerConfig(hydrogen_steps=5, heavy_steps=10)
    ).refine(_solved("F/C=C/F"))
    assert result.accepted
    assert result.metadata["specified_ez_bonds"] == 1
    assert result.metadata["ez_violations_after"] <= result.metadata[
        "ez_violations_before"
    ]
