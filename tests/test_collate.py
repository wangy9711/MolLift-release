"""Tests for batch collate padding."""

import numpy as np
import torch
from mollift_generator.data.collate import collate_molecules
from mollift_generator.data.featurize import featurize_mol, mol_from_smiles
from mollift_generator.data.projection import synthetic_projection


def _make_sample(smiles: str):
    mol = mol_from_smiles(smiles)
    rng = np.random.default_rng(0)
    x3d = rng.normal(size=(mol.GetNumAtoms(), 3))
    x3d, pos_2d = synthetic_projection(x3d, rng=rng)
    feat = featurize_mol(mol)
    return {
        "smiles": smiles,
        "atom_types": feat["atom_types"],
        "atom_features": feat["atom_features"],
        "bond_index": feat["bond_index"],
        "bond_features": feat["bond_features"],
        "bond_features_dense": feat["bond_features_dense"],
        "shortest_path": feat["shortest_path"],
        "x3d": x3d,
        "pos_2d": pos_2d,
        "n_atoms": mol.GetNumAtoms(),
    }


def test_collate_padding():
    batch = collate_molecules([_make_sample("C"), _make_sample("CCO")], max_atoms=10)
    assert batch["atom_features"].shape[0] == 2
    assert batch["atom_features"].shape[1] == 10
    assert batch["node_mask"].dtype == torch.bool
    assert batch["node_mask"][0].sum() == 1
    assert batch["node_mask"][1].sum() == 3
    assert batch["x3d"][0, 1:].abs().sum() == 0
