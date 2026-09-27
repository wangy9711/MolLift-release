"""Validated atom-aligned inputs and portable output serialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from rdkit import Chem

from mollift_solver import MoleculeInput


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")


def validate_molecule(molecule: MoleculeInput) -> MoleculeInput:
    molecule.validate()
    mol = molecule.to_rdkit_mol()
    if mol.GetNumAtoms() < 2 or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("Provide a single connected molecule with at least two atoms")
    mask = np.asarray(molecule.coords_2d_mask)
    if mask.dtype != bool or mask.shape != (mol.GetNumAtoms(),):
        raise ValueError("observed_mask must contain one boolean per modeled atom")
    if not mask.any():
        raise ValueError("At least one observed atom is required for projection-conditioned inference")
    if not np.isfinite(molecule.coords_2d[mask]).all():
        raise ValueError("Observed coordinates must be finite")
    return molecule


def load_input(path: str | Path) -> MoleculeInput:
    data = json.loads(Path(path).read_text())
    if data.get("units") != "angstrom":
        raise ValueError("Input units must be 'angstrom'; use --annotation for pixels")
    smiles = data["smiles"]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid input SMILES")
    representation = data.get("atom_representation")
    if representation not in {"heavy", "explicit_h"}:
        raise ValueError("Specify atom_representation: heavy or explicit_h")
    if representation == "explicit_h":
        mol = Chem.AddHs(mol)
    expected = [a.GetAtomicNum() for a in mol.GetAtoms()]
    if data.get("atomic_numbers") != expected:
        raise ValueError("atomic_numbers must match the exact parsed SMILES / AddHs atom order")
    n = mol.GetNumAtoms()
    raw_mask = data.get("observed_mask")
    if not isinstance(raw_mask, list) or any(type(x) is not bool for x in raw_mask):
        raise ValueError("observed_mask must be a JSON list of booleans")
    mask = np.asarray(raw_mask, dtype=bool)
    xy = np.asarray(data["coords_2d"], dtype=np.float64)
    if xy.shape != (n, 2) or mask.shape != (n,):
        raise ValueError(f"Expected coords_2d ({n}, 2) and observed_mask ({n},)")
    if not np.isfinite(xy[mask]).all():
        raise ValueError("Observed coordinates must be finite")
    xy[~mask] = 0.0  # Missing positions never provide hidden observations.
    centering = data.get("centering", "provided")
    offset = np.zeros(2)
    if centering == "observed":
        if not mask.any():
            raise ValueError("No observed atoms to center")
        offset = xy[mask].mean(0)
        xy[mask] -= offset
    elif centering != "provided":
        raise ValueError("centering must be 'provided' or 'observed'")
    return validate_molecule(MoleculeInput(
        smiles=smiles, coords_2d=xy, coords_2d_mask=mask, atomic_numbers=expected,
        metadata={"input_format": "angstrom-json", "atom_representation": representation,
                  "centering": centering, "subtracted_xy_origin": offset.tolist()},
    ))


def load_annotation(path: str | Path, pixels_per_angstrom: float | None = None) -> MoleculeInput:
    # The schema/export modules have no Qt dependency.
    from mollift_annotator.schema import load_record
    from mollift_annotator.export import annotations_to_pixel_array
    from mollift.calibration import calibrate_pixels_to_plane
    from mollift_solver.topology import build_topology

    record = load_record(path)
    pixels, mask = annotations_to_pixel_array(record)
    if not mask.any() or not np.isfinite(pixels[mask]).all():
        raise ValueError("Annotation must contain finite visible pixel coordinates")
    mol = Chem.MolFromSmiles(record.smiles)
    for ann in record.annotations:
        if ann.symbol != mol.GetAtomWithIdx(ann.atom_index).GetSymbol():
            raise ValueError("Annotation symbol does not match its SMILES atom index")
    if pixels_per_angstrom is not None and (
        not np.isfinite(pixels_per_angstrom) or pixels_per_angstrom <= 0
    ):
        raise ValueError("pixels_per_angstrom must be positive and finite")
    calibration = calibrate_pixels_to_plane(
        pixels, mask, build_topology(record.smiles), pixels_per_angstrom,
    )
    xy = np.zeros((mol.GetNumAtoms(), 2))
    xy[mask] = calibration.coords_2d[mask]
    return validate_molecule(MoleculeInput(
        smiles=record.smiles, coords_2d=xy, coords_2d_mask=mask,
        atomic_numbers=[a.GetAtomicNum() for a in mol.GetAtoms()],
        metadata={"input_format": "annotator-json", "atom_representation": "heavy",
                  "centering": "observed", "image_y_flipped": True,
                  "pixels_per_angstrom": calibration.pixels_per_angstrom,
                  "calibration_bond_pairs": calibration.n_bond_pairs_used},
    ))


def write_sdf(path: Path, molecule: MoleculeInput, candidates: np.ndarray, indices=None):
    mol = molecule.to_rdkit_mol()
    indices = range(len(candidates)) if indices is None else indices
    with Chem.SDWriter(str(path)) as writer:
        for index in indices:
            coords = candidates[index]
            if not np.isfinite(coords).all():
                continue  # NPZ/JSON retain failed candidates at their original indices.
            copy = Chem.Mol(mol)
            conf = Chem.Conformer(copy.GetNumAtoms())
            for i, xyz in enumerate(coords):
                conf.SetAtomPosition(i, tuple(map(float, xyz)))
            copy.RemoveAllConformers()
            copy.AddConformer(conf, assignId=True)
            copy.SetIntProp("candidate_index", int(index))
            copy.SetProp("coordinate_frame", "input_observation_frame")
            writer.write(copy)
