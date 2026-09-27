"""RDKit molecular featurization."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import ChiralType
from rdkit.Chem.rdchem import ValenceType

ATOMIC_TYPES_QM9 = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
# Fixed vocabulary for datasets with broader chemistry.  The final one-hot slot from
# ``one_hot_encoding`` remains the unknown bucket, so uncommon elements keep a stable
# feature width instead of making it depend on the atoms present in each molecule.
ATOMIC_SYMBOLS_GEOM = [
    "H",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Si",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
]
BONDS = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
BOND_FEATURE_DIM = len(BONDS) + 1  # + misc


def one_hot_encoding(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def get_node_attr_from_mol(mol: Chem.Mol, dataset: str = "qm9") -> np.ndarray:
    all_node_attr = []
    ring = mol.GetRingInfo()
    types = list(ATOMIC_TYPES_QM9.keys()) if dataset == "qm9" else ATOMIC_SYMBOLS_GEOM
    for i, atom in enumerate(mol.GetAtoms()):
        node_attr = []
        node_attr.extend(
            one_hot_encoding(
                atom.GetChiralTag(),
                [
                    ChiralType.CHI_TETRAHEDRAL_CW,
                    ChiralType.CHI_TETRAHEDRAL_CCW,
                    ChiralType.CHI_UNSPECIFIED,
                    ChiralType.CHI_OTHER,
                ],
            )
        )
        node_attr.extend(one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]))
        node_attr.extend(one_hot_encoding(atom.GetNumRadicalElectrons(), [0, 1, 2, 3, 4]))
        if dataset == "qm9":
            # Preserve the existing QM9 feature width and behavior exactly.
            node_attr.extend(one_hot_encoding(atom.GetSymbol(), types)[:-1])
        else:
            node_attr.extend(one_hot_encoding(atom.GetSymbol(), types))
        node_attr.append(int(atom.GetIsAromatic()))
        node_attr.extend(one_hot_encoding(atom.GetTotalDegree(), [0, 1, 2, 3, 4]))
        node_attr.extend(
            one_hot_encoding(
                atom.GetHybridization(),
                [
                    Chem.rdchem.HybridizationType.SP,
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,
                    Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2,
                ],
            )
        )
        node_attr.extend(one_hot_encoding(atom.GetValence(ValenceType.IMPLICIT), [0, 1, 2, 3, 4]))
        node_attr.extend(one_hot_encoding(atom.GetFormalCharge(), [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]))
        node_attr.extend(
            [
                int(ring.IsAtomInRingOfSize(i, s))
                for s in (3, 4, 5, 6, 7, 8)
            ]
        )
        node_attr.extend(one_hot_encoding(int(ring.NumAtomRings(i)), [0, 1, 2, 3]))
        all_node_attr.append(node_attr)
    return np.array(all_node_attr, dtype=np.float32)


def get_atom_types_from_mol(mol: Chem.Mol) -> np.ndarray:
    return np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int64)


def get_dense_bond_features(mol: Chem.Mol) -> Tuple[np.ndarray, np.ndarray]:
    """Return bond_adj [N,N] and bond_feat [N,N,D_bond]."""
    n = mol.GetNumAtoms()
    bonds_values = list(BONDS.values())
    bond_adj = np.zeros((n, n), dtype=np.float32)
    bond_feat = np.zeros((n, n, BOND_FEATURE_DIM), dtype=np.float32)
    misc = one_hot_encoding(-100, bonds_values)
    for i in range(n):
        for j in range(n):
            bond_feat[i, j] = misc

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = one_hot_encoding(BONDS.get(bond.GetBondType(), -100), bonds_values)
        bond_adj[i, j] = 1.0
        bond_adj[j, i] = 1.0
        bond_feat[i, j] = feat
        bond_feat[j, i] = feat
    return bond_adj, bond_feat


_UNREACHABLE_HOPS = 510


def floyd_warshall(adj: np.ndarray, max_hop: int | None = None) -> np.ndarray:
    """Shortest path hops; unreachable -> 510 (dit_mc algos.pyx convention)."""
    del max_hop  # kept for API compatibility; no clipping applied
    n = adj.shape[0]
    dist = np.full((n, n), _UNREACHABLE_HOPS, dtype=np.int64)
    np.fill_diagonal(dist, 0)
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                dist[i, j] = 1
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
    return dist


def featurize_mol(mol: Chem.Mol, max_hop: int = 8, dataset: str = "qm9") -> Dict[str, np.ndarray]:
    atom_types = get_atom_types_from_mol(mol)
    atom_features = get_node_attr_from_mol(mol, dataset=dataset)
    bond_adj, bond_features = get_dense_bond_features(mol)
    shortest_path = floyd_warshall(bond_adj, max_hop=max_hop)

    bond_index = []
    bond_feat_list = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features[i, j]
        bond_index.append([i, j])
        bond_index.append([j, i])
        bond_feat_list.append(feat)
        bond_feat_list.append(feat)

    if bond_index:
        bond_index = np.array(bond_index, dtype=np.int64).T
        bond_feat_edges = np.stack(bond_feat_list, axis=0)
    else:
        bond_index = np.zeros((2, 0), dtype=np.int64)
        bond_feat_edges = np.zeros((0, BOND_FEATURE_DIM), dtype=np.float32)

    return {
        "atom_types": atom_types,
        "atom_features": atom_features,
        "bond_index": bond_index,
        "bond_features": bond_feat_edges,
        "bond_adj": bond_adj,
        "bond_features_dense": bond_features,
        "shortest_path": shortest_path,
    }


def mol_from_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol


def filter_mol(mol: Chem.Mol, smiles: str) -> bool:
    if mol is None or "." in smiles:
        return False
    for atom in mol.GetAtoms():
        if atom.GetDegree() > 4:
            return False
    return True
