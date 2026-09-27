"""Read and convert the trusted RDKit pickle release of GEOM."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem

from mollift_generator.data.featurize import featurize_mol, filter_mol
from mollift_generator.data.geom_cache import (
    GEOM_CACHE_VERSION,
    GEOMRecord,
    cache_path_for,
    molecule_id,
)
from mollift_generator.training.flow_utils import build_prior_graph


def load_geom_pickle(path: str | Path) -> dict[str, Any]:
    """Load one GEOM release file.

    Pickle can execute code. This function is only intended for the trusted, locally
    downloaded GEOM release and must not be used on untrusted uploads.
    """
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "conformers" not in payload:
        raise ValueError(f"Not a GEOM molecule pickle: {path}")
    return payload


def _conformer_coords(mol: Chem.Mol) -> np.ndarray:
    if mol.GetNumConformers() < 1:
        raise ValueError("RDKit molecule has no conformer")
    coords = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
    if coords.shape != (mol.GetNumAtoms(), 3) or not np.isfinite(coords).all():
        raise ValueError("invalid conformer coordinates")
    return coords


def _prior_arrays(bond_features_dense: np.ndarray, n_atoms: int) -> tuple[np.ndarray, np.ndarray]:
    bond = torch.from_numpy(bond_features_dense).unsqueeze(0).float()
    node_mask = torch.ones(1, n_atoms, dtype=torch.bool)
    prior = build_prior_graph(bond, node_mask)
    return (
        prior.node_attr[0, :n_atoms].numpy().astype(np.float32),
        prior.evecs[0, :n_atoms, :n_atoms].numpy().astype(np.float32),
    )


def _optional_float(entry: dict[str, Any], key: str) -> float:
    value = entry.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def convert_geom_pickle(
    source_path: str | Path,
    cache_root: str | Path,
    split: str,
    *,
    max_atoms: int | None = None,
    min_conformers: int = 1,
    max_conformers: int | None = None,
    max_hop: int = 8,
    overwrite: bool = False,
) -> GEOMRecord:
    """Convert one GEOM molecule to the compact molecule-level training cache."""
    source_path = Path(source_path)
    raw = load_geom_pickle(source_path)
    conformer_entries = raw.get("conformers") or []
    if not conformer_entries:
        raise ValueError("no conformers")

    first_mol = conformer_entries[0].get("rd_mol")
    if not isinstance(first_mol, Chem.Mol):
        raise ValueError("first conformer has no RDKit molecule")

    smiles = str(raw.get("smiles") or Chem.MolToSmiles(Chem.RemoveHs(first_mol), isomericSmiles=True))
    canonical = Chem.MolToSmiles(Chem.RemoveHs(first_mol), canonical=True, isomericSmiles=True)
    if "." in canonical or not filter_mol(first_mol, canonical):
        raise ValueError("disconnected or unsupported molecule")

    n_atoms = first_mol.GetNumAtoms()
    if max_atoms is not None and n_atoms > max_atoms:
        raise ValueError(f"too many atoms: {n_atoms} > {max_atoms}")

    key = molecule_id(canonical)
    out_path = cache_path_for(cache_root, key)
    if out_path.exists() and not overwrite:
        cached = torch.load(out_path, map_location="cpu", weights_only=False)
        if int(cached.get("cache_version", -1)) == GEOM_CACHE_VERSION:
            return GEOMRecord(
                key,
                split,
                str(out_path.resolve()),
                str(cached["smiles"]),
                str(source_path.resolve()),
                int(cached["n_atoms"]),
                int(cached["conformer_coords"].shape[0]),
                int(cached["atom_features"].shape[-1]),
            )

    feat = featurize_mol(first_mol, max_hop=max_hop, dataset="geom")
    prior_node_attr, prior_evecs = _prior_arrays(feat["bond_features_dense"], n_atoms)

    coords: list[np.ndarray] = []
    relative_energy: list[float] = []
    total_energy: list[float] = []
    boltzmann_weight: list[float] = []
    degeneracy: list[float] = []
    geom_ids: list[str] = []
    reference_atomic_numbers = np.asarray(
        [atom.GetAtomicNum() for atom in first_mol.GetAtoms()],
        dtype=np.int16,
    )
    limit = len(conformer_entries) if max_conformers is None else max_conformers
    for entry in conformer_entries[:limit]:
        mol = entry.get("rd_mol")
        if not isinstance(mol, Chem.Mol) or mol.GetNumAtoms() != n_atoms:
            continue
        atomic_numbers = np.asarray(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()],
            dtype=np.int16,
        )
        if not np.array_equal(atomic_numbers, reference_atomic_numbers):
            # Coordinate supervision requires identical atom identity and order.
            continue
        try:
            xyz = _conformer_coords(mol)
        except ValueError:
            continue
        coords.append(xyz)
        relative_energy.append(_optional_float(entry, "relativeenergy"))
        total_energy.append(_optional_float(entry, "totalenergy"))
        boltzmann_weight.append(_optional_float(entry, "boltzmannweight"))
        degeneracy.append(_optional_float(entry, "degeneracy"))
        geom_ids.append(str(entry.get("geom_id", "")))

    if len(coords) < min_conformers:
        raise ValueError(f"only {len(coords)} valid conformers; need {min_conformers}")

    payload = {
        "cache_version": GEOM_CACHE_VERSION,
        "molecule_id": key,
        "source_path": str(source_path.resolve()),
        "smiles": smiles,
        "canonical_smiles": canonical,
        "n_atoms": n_atoms,
        "atom_types": feat["atom_types"],
        "atom_features": feat["atom_features"],
        "bond_index": feat["bond_index"],
        "bond_features": feat["bond_features"],
        "bond_features_dense": feat["bond_features_dense"],
        "shortest_path": feat["shortest_path"],
        "prior_node_attr": prior_node_attr,
        "prior_evecs": prior_evecs,
        "conformer_coords": np.stack(coords).astype(np.float32),
        "relative_energy": np.asarray(relative_energy, dtype=np.float32),
        "total_energy": np.asarray(total_energy, dtype=np.float64),
        "boltzmann_weight": np.asarray(boltzmann_weight, dtype=np.float32),
        "degeneracy": np.asarray(degeneracy, dtype=np.float32),
        "geom_ids": geom_ids,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(f".tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, out_path)

    return GEOMRecord(
        key,
        split,  # type: ignore[arg-type]
        str(out_path.resolve()),
        smiles,
        str(source_path.resolve()),
        n_atoms,
        len(coords),
        int(feat["atom_features"].shape[-1]),
    )
