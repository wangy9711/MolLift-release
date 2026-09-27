import json
from pathlib import Path

import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

from mollift.io import load_input, load_annotation
from mollift.selection import select_candidate
from mollift_solver import SolveConfig
from mollift_solver.optimize import torsion_loss, optimize_multi_start
from mollift_solver.projection import Plane2DFrame
from mollift_solver.topology import build_topology


def test_missing_positions_never_become_conditions(tmp_path):
    payload = {"smiles": "CCO", "units": "angstrom", "atom_representation": "heavy",
               "atomic_numbers": [6, 6, 8], "coords_2d": [[0, 0], [1, 0], [None, None]],
               "observed_mask": [True, True, False]}
    p = tmp_path / "input.json"
    p.write_text(json.dumps(payload))
    first = load_input(p)
    payload["coords_2d"][2] = [999, -123]
    p.write_text(json.dumps(payload))
    second = load_input(p)
    assert np.array_equal(first.coords_2d, second.coords_2d)
    payload["atomic_numbers"] = [8, 6, 6]
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="atom order"):
        load_input(p)


def test_no_observations_or_nonfinite_observations_rejected(tmp_path):
    p = tmp_path / "input.json"
    payload = {"smiles": "CC", "units": "angstrom", "atom_representation": "heavy",
               "atomic_numbers": [6, 6], "coords_2d": [[0, 0], [1, 0]],
               "observed_mask": [False, False]}
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="At least one"):
        load_input(p)
    payload["observed_mask"][0] = True
    payload["coords_2d"][0][0] = None
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="finite"):
        load_input(p)


def test_selector_uses_projection_window_then_comparable_energy():
    coords = np.zeros((4, 2, 3))
    rows = [
        {"accepted": True, "energy": 10., "force_field": "MMFF94s", "projection_rmse": .1},
        {"accepted": True, "energy": 1., "force_field": "MMFF94s", "projection_rmse": .15},
        {"accepted": True, "energy": -900., "force_field": "MMFF94s", "projection_rmse": .3},
        {"accepted": True, "energy": -1000., "force_field": "UFF", "projection_rmse": .1},
    ]
    assert select_candidate("GR", coords, refiner_rows=rows) == 1
    assert select_candidate("GSR", coords, refiner_rows=rows) == 1
    assert select_candidate("G", coords) == 0
    assert select_candidate("GS", coords, objectives=[5, 3, 1, 2]) == 2
    for row in rows:
        row["accepted"] = False
    assert select_candidate("GR", coords, refiner_rows=rows) is None


@pytest.mark.parametrize("pipeline", [None, "G", "GR", "GS", "GSR"])
def test_cli_pipeline_names_and_default(monkeypatch, tmp_path, pipeline):
    from mollift import cli, predict

    selected = []
    checkpoint = tmp_path / "model.pt"
    monkeypatch.setattr(cli, "checkpoint_for", lambda args: (checkpoint, "mollift-drugs"))
    monkeypatch.setattr(predict, "run", lambda args, path, name: selected.append(args.pipeline))
    command = ["predict", "--input", str(tmp_path / "input.json"),
               "--output", str(tmp_path / "output")]
    if pipeline is not None:
        command += ["--pipeline", pipeline]
    cli.main(command)
    assert selected == [pipeline or "GR"]


@pytest.mark.parametrize("smiles", ["CCCC", "CCOC", "CCNC"])
def test_solver_torsion_matches_rdkit_and_has_correct_gradients(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.MMFFGetMoleculeProperties(mol).GetMMFFTorsionParams(mol, 0, 1, 2, 3)[1:]
    q = torch.tensor([[0, 1, 2, 3]])
    p = torch.tensor([params], dtype=torch.float64)
    for degrees in [0, 30, 60, 90, 120, 150, 180, 240, 300]:
        angle = np.deg2rad(degrees)
        xyz = np.array([[0., 1., 0.], [0., 0., 0.], [1., 0., 0.],
                        [1., np.cos(angle), np.sin(angle)]])
        conf = Chem.Conformer(4)
        for i, v in enumerate(xyz):
            conf.SetAtomPosition(i, tuple(v))
        phi = rdMolTransforms.GetDihedralRad(conf, 0, 1, 2, 3)
        v1, v2, v3 = params
        expected = .5 * (v1*(1+np.cos(phi)) + v2*(1-np.cos(2*phi)) + v3*(1+np.cos(3*phi)))
        assert float(torsion_loss(torch.tensor(xyz), q, p)) == pytest.approx(expected, abs=1e-10)
    xyz = torch.tensor([[0., 1., .2], [0., 0., 0.], [1., 0., 0.], [1.1, .4, .8]],
                       dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: torsion_loss(x, q, p), (xyz,))


def test_solver_returns_minimum_final_objective_and_enters_second_stage():
    torch.set_num_threads(1)
    topo = build_topology("CCCC")
    initial = list(np.random.default_rng(17).normal(size=(2, 4, 3)))
    projection = np.array([[0., 0.], [1.5, 0.], [2.4, 1.], [3.8, .7]])
    result = optimize_multi_start(initial, projection, topo, Plane2DFrame.default_xy_plane(),
                                 SolveConfig(steps=50, dtype="float64"))
    history = np.asarray(result.diagnostics["full_objective_trajectory"])
    assert result.diagnostics["actual_steps"] >= 45
    assert result.diagnostics["entered_stage2"]
    assert np.allclose(result.candidate_final_losses, history.min(0), atol=1e-10)


def test_annotation_import_does_not_require_image_or_reference():
    path = Path(__file__).resolve().parents[1] / "examples/peptide_annotation.json"
    molecule = load_annotation(path, 40)
    assert molecule.metadata["pixels_per_angstrom"] == 40
    assert np.allclose(molecule.coords_2d[molecule.coords_2d_mask].mean(0), 0)
    assert np.isfinite(molecule.coords_2d).all()
