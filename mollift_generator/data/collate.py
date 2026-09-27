"""Dense padding collate for variable-size molecules."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch


def _pad_array(arr: np.ndarray, shape: tuple, fill=0) -> np.ndarray:
    out = np.full(shape, fill, dtype=arr.dtype)
    slices = tuple(slice(0, s) for s in arr.shape)
    out[slices] = arr
    return out


def collate_molecules(batch: List[Dict[str, Any]], max_atoms: int | None = None) -> Dict[str, torch.Tensor]:
    if max_atoms is None:
        max_atoms = max(int(b["n_atoms"]) for b in batch)

    bsz = len(batch)
    d_atom = batch[0]["atom_features"].shape[-1]
    d_bond = batch[0]["bond_features_dense"].shape[-1]
    has_prior = "prior_node_attr" in batch[0] and "prior_evecs" in batch[0]
    has_chirality = "chirality_feat" in batch[0]
    has_experimental_mask = any("experimental_atom_mask" in sample for sample in batch)

    atom_types = np.zeros((bsz, max_atoms), dtype=np.int64)
    atom_features = np.zeros((bsz, max_atoms, d_atom), dtype=np.float32)
    shortest_path = np.full((bsz, max_atoms, max_atoms), fill_value=999, dtype=np.int64)
    x3d = np.zeros((bsz, max_atoms, 3), dtype=np.float32)
    pos_2d = np.zeros((bsz, max_atoms, 2), dtype=np.float32)
    bond_features_dense = np.zeros((bsz, max_atoms, max_atoms, d_bond), dtype=np.float32)
    node_mask = np.zeros((bsz, max_atoms), dtype=bool)
    sample_keys = np.zeros(bsz, dtype=np.int64)
    prior_node_attr = np.zeros((bsz, max_atoms), dtype=np.float32) if has_prior else None
    prior_evecs = np.zeros((bsz, max_atoms, max_atoms), dtype=np.float32) if has_prior else None
    d_chir = batch[0]["chirality_feat"].shape[-1] if has_chirality else 0
    chirality_feat = np.zeros((bsz, max_atoms, d_chir), dtype=np.float32) if has_chirality else None
    experimental_atom_mask = np.zeros((bsz, max_atoms), dtype=bool) if has_experimental_mask else None
    cap_atom_mask = np.zeros((bsz, max_atoms), dtype=bool) if has_experimental_mask else None

    for i, sample in enumerate(batch):
        n = int(sample["n_atoms"])
        node_mask[i, :n] = True
        sample_keys[i] = int(sample.get("sample_key", i))
        atom_types[i, :n] = sample["atom_types"]
        atom_features[i, :n] = sample["atom_features"]
        shortest_path[i, :n, :n] = sample["shortest_path"]
        x3d[i, :n] = sample["x3d"]
        pos_2d[i, :n] = sample["pos_2d"]
        bond_features_dense[i, :n, :n] = sample["bond_features_dense"]
        if has_prior:
            prior_node_attr[i, :n] = sample["prior_node_attr"]
            prior_evecs[i, :n, :n] = sample["prior_evecs"]
        if has_chirality:
            chirality_feat[i, :n] = sample["chirality_feat"]
        if has_experimental_mask and "experimental_atom_mask" in sample:
            experimental_atom_mask[i, :n] = sample["experimental_atom_mask"]
            cap_atom_mask[i, :n] = sample["cap_atom_mask"]

    pair_mask = node_mask[:, :, None] & node_mask[:, None, :]

    out = {
        "smiles": [b["smiles"] for b in batch],
        "atom_types": torch.from_numpy(atom_types),
        "atom_features": torch.from_numpy(atom_features),
        "shortest_path": torch.from_numpy(shortest_path),
        "bond_features_dense": torch.from_numpy(bond_features_dense),
        "x3d": torch.from_numpy(x3d),
        "pos_2d": torch.from_numpy(pos_2d),
        "node_mask": torch.from_numpy(node_mask),
        "pair_mask": torch.from_numpy(pair_mask),
        "sample_keys": torch.from_numpy(sample_keys),
    }
    if has_prior:
        out["prior_node_attr"] = torch.from_numpy(prior_node_attr)
        out["prior_evecs"] = torch.from_numpy(prior_evecs)
    if has_chirality:
        out["chirality_feat"] = torch.from_numpy(chirality_feat)
    if has_experimental_mask:
        out["experimental_atom_mask"] = torch.from_numpy(experimental_atom_mask)
        out["cap_atom_mask"] = torch.from_numpy(cap_atom_mask)
    if any("data_source" in sample for sample in batch):
        out["data_source"] = [sample.get("data_source", "unknown") for sample in batch]
    return out


def collate_molecules_capped(
    batch: List[Dict[str, Any]],
    max_atoms_cap: int | None = None,
) -> Dict[str, torch.Tensor]:
    """Pad to the largest molecule in the batch, optionally capped by ``max_atoms_cap``."""
    dynamic = max(int(b["n_atoms"]) for b in batch)
    if max_atoms_cap is not None:
        dynamic = min(dynamic, max_atoms_cap)
    return collate_molecules(batch, max_atoms=dynamic)
