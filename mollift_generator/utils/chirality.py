"""Chirality correction ported from dit_mc generate_confs.py."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import FindMolChiralCenters
from rdkit.Chem.rdchem import ChiralType

CHIRALITY = {
    ChiralType.CHI_TETRAHEDRAL_CW: -1.0,
    ChiralType.CHI_TETRAHEDRAL_CCW: 1.0,
    ChiralType.CHI_UNSPECIFIED: 0.0,
    ChiralType.CHI_OTHER: 0.0,
}


def get_chiral_centers_from_mol(mol) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return chiral center indices, flattened neighbor indices, and tags."""
    chiral_index = []
    chiral_nbr_index = []
    chiral_tag = []
    for i, atom in enumerate(mol.GetAtoms()):
        tag = CHIRALITY.get(atom.GetChiralTag(), 0.0)
        if tag != 0 and len(atom.GetNeighbors()) == 4:
            chiral_index.append(i)
            chiral_nbr_index.extend(n.GetIdx() for n in atom.GetNeighbors())
            chiral_tag.append(tag)
    if not chiral_index:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float32),
        )
    return (
        np.array(chiral_index, dtype=np.int32),
        np.array(chiral_nbr_index, dtype=np.int32),
        np.array(chiral_tag, dtype=np.float32),
    )


def signed_volume(local_coords: np.ndarray) -> np.ndarray:
    """Signed volume for ordered neighbor coordinates [C, 4, 3]."""
    v1 = local_coords[:, 0] - local_coords[:, 3]
    v2 = local_coords[:, 1] - local_coords[:, 3]
    v3 = local_coords[:, 2] - local_coords[:, 3]
    cp = np.cross(v2, v3)
    return np.sign(np.sum(v1 * cp, axis=-1))


def switch_parity_of_pos(
    pos: np.ndarray,
    chiral_nbr_index: np.ndarray,
    chiral_tag: np.ndarray,
) -> np.ndarray:
    """Flip z-axis if all chiral centers have wrong parity (dit_mc behavior)."""
    if chiral_nbr_index.size == 0:
        return pos
    nbr = chiral_nbr_index.reshape(-1, 4)
    sv = signed_volume(pos[nbr])
    z_flip = sv * chiral_tag
    node_factor = -1.0 if (z_flip == -1.0).all() else 1.0
    flip_mat = np.diag([1.0, 1.0, node_factor])
    return pos @ flip_mat.T


def correct_chirality(pos: np.ndarray, mol) -> np.ndarray:
    """Apply chirality correction to coordinates [N, 3]."""
    _, chiral_nbr_index, chiral_tag = get_chiral_centers_from_mol(mol)
    return switch_parity_of_pos(pos, chiral_nbr_index, chiral_tag)


REFLECT_Z = np.array([1.0, 1.0, -1.0])


def _decision_from_parity(z_flip: np.ndarray) -> str:
    """Classify parity products from signed volumes and target chiral tags.

    Negative entries indicate inverted centers. Reflect only when all centers
    are inverted; mixed signs require local corrections.
    """
    n = int(z_flip.size)
    if n == 0:
        return "achiral"
    n_mismatch = int((z_flip < 0).sum())
    if n_mismatch == 0:
        return "keep"
    if n_mismatch == n:
        return "reflect"
    return "diastereomer"


def chirality_correct(pos: np.ndarray, mol) -> tuple[np.ndarray, dict]:
    """Correct a global z reflection using tetrahedral tags from isomeric SMILES.

    Only tagged centers with four explicit neighbors are checked. Mixed parity
    mismatches are reported without changing coordinates; global reflection
    cannot repair local stereochemical errors.

    Return ``(corrected_pos, info)`` with decision, n_centers and n_mismatch.
    Decisions are ``achiral``, ``keep``, ``reflect`` or ``diastereomer``; ``achiral``
    means that no eligible tagged centers were found.
    """
    _, chiral_nbr_index, chiral_tag = get_chiral_centers_from_mol(mol)
    n_centers = int(chiral_tag.size)
    if n_centers == 0:
        return pos, {"decision": "achiral", "n_centers": 0, "n_mismatch": 0}
    nbr = chiral_nbr_index.reshape(-1, 4)
    z_flip = signed_volume(pos[nbr]) * chiral_tag
    n_mismatch = int((z_flip < 0).sum())
    decision = _decision_from_parity(z_flip)
    corrected = pos * REFLECT_Z if decision == "reflect" else pos
    return corrected, {
        "decision": decision,
        "n_centers": n_centers,
        "n_mismatch": n_mismatch,
    }


# Reserved feature width retained for checkpoint-compatible model construction.
CHIRALITY_FEATURE_DIM = 3
