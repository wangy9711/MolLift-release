"""Build molecular topology and ideal geometry targets from SMILES."""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdForceFieldHelpers

from mollift_solver.types import MoleculeTopology

# Fallback ideal bond lengths (Angstrom)
_BOND_LENGTH_FALLBACK = {
    Chem.rdchem.BondType.SINGLE: 1.54,
    Chem.rdchem.BondType.DOUBLE: 1.34,
    Chem.rdchem.BondType.TRIPLE: 1.20,
    Chem.rdchem.BondType.AROMATIC: 1.40,
}

# Typical angles (radians)
_ANGLE_SP3 = math.radians(109.5)
_ANGLE_SP2 = math.radians(120.0)
_ANGLE_SP = math.radians(180.0)
_RING_ANGLE = {
    3: math.radians(60.0),
    4: math.radians(90.0),
    5: math.radians(108.0),
    6: math.radians(120.0),
}


def mol_from_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return mol


def validate_coords_2d(mol: Chem.Mol, coords_2d: np.ndarray) -> None:
    n = mol.GetNumAtoms()
    coords_2d = np.asarray(coords_2d)
    if coords_2d.ndim != 2 or coords_2d.shape[1] != 2:
        raise ValueError(f"coords_2d must be (N, 2), got {coords_2d.shape}")
    if coords_2d.shape[0] != n:
        raise ValueError(
            f"coords_2d has {coords_2d.shape[0]} rows but mol has {n} atoms "
            "(must match RDKit atom order)"
        )


def validate_coords_2d_partial(
    mol: Chem.Mol,
    coords_2d: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Validate partial 2D observations; mask True = observed atom."""
    n = mol.GetNumAtoms()
    coords_2d = np.asarray(coords_2d)
    mask = np.asarray(mask, dtype=bool)
    if coords_2d.ndim != 2 or coords_2d.shape[1] != 2:
        raise ValueError(f"coords_2d must be (N, 2), got {coords_2d.shape}")
    if coords_2d.shape[0] != n:
        raise ValueError(
            f"coords_2d has {coords_2d.shape[0]} rows but mol has {n} atoms "
            "(must match RDKit atom order)"
        )
    if mask.shape != (n,):
        raise ValueError(f"coords_2d_mask must be ({n},), got {mask.shape}")
    if mask.sum() < 2:
        raise ValueError("At least 2 atoms must be observed (coords_2d_mask.sum() >= 2)")
    if np.any(np.isnan(coords_2d[mask])):
        raise ValueError("Observed coords_2d rows (mask=True) must not contain NaN")


def _ideal_bond_length(bond: Chem.Bond) -> float:
    """Element-aware covalent-radius fallback when force-field typing fails."""
    bt = bond.GetBondType()
    pt = Chem.GetPeriodicTable()
    base = pt.GetRcovalent(bond.GetBeginAtom().GetAtomicNum()) + pt.GetRcovalent(
        bond.GetEndAtom().GetAtomicNum()
    )
    correction = {
        Chem.rdchem.BondType.SINGLE: 0.00,
        Chem.rdchem.BondType.DOUBLE: -0.12,
        Chem.rdchem.BondType.TRIPLE: -0.22,
        Chem.rdchem.BondType.AROMATIC: -0.08,
    }.get(bt, 0.0)
    value = float(base + correction)
    return value if 0.8 <= value <= 2.5 else _BOND_LENGTH_FALLBACK.get(bt, 1.5)


def _collect_bonds(
    mol: Chem.Mol, mmff_props=None, ff_mol: Chem.Mol | None = None
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    edges: List[Tuple[int, int]] = []
    lengths: List[float] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append((i, j))
        params = None
        if mmff_props is not None:
            params = mmff_props.GetMMFFBondStretchParams(ff_mol or mol, i, j)
        if params is not None:
            lengths.append(float(params[2]))
            continue
        uff = rdForceFieldHelpers.GetUFFBondStretchParams(mol, i, j)
        lengths.append(float(uff[1]) if uff is not None else _ideal_bond_length(bond))
    return edges, np.array(lengths, dtype=np.float64)


def _collect_angles(
    mol: Chem.Mol, mmff_props=None, ff_mol: Chem.Mol | None = None
) -> Tuple[List[Tuple[int, int, int]], List[float]]:
    """Angle triplets (i, j, k) with j the vertex."""
    triplets: List[Tuple[int, int, int]] = []
    targets: List[float] = []
    seen = set()

    for atom_j in mol.GetAtoms():
        j = atom_j.GetIdx()
        neighbors = [n.GetIdx() for n in atom_j.GetNeighbors()]
        if len(neighbors) < 2:
            continue

        target = _guess_angle_target(mol, j, neighbors)

        for a in range(len(neighbors)):
            for b in range(a + 1, len(neighbors)):
                i, k = neighbors[a], neighbors[b]
                key = (min(i, k), j, max(i, k))
                if key in seen:
                    continue
                seen.add(key)
                triplets.append((i, j, k))
                params = None
                if mmff_props is not None:
                    params = mmff_props.GetMMFFAngleBendParams(ff_mol or mol, i, j, k)
                if params is not None:
                    targets.append(math.radians(float(params[2])))
                else:
                    uff = rdForceFieldHelpers.GetUFFAngleBendParams(mol, i, j, k)
                    targets.append(math.radians(float(uff[1])) if uff is not None else target)

    return triplets, targets


def _guess_angle_target(mol: Chem.Mol, j: int, neighbors: List[int]) -> float:
    atom = mol.GetAtomWithIdx(j)
    hyb = atom.GetHybridization()

    if atom.GetIsAromatic() or hyb == Chem.rdchem.HybridizationType.SP2:
        return _ANGLE_SP2
    if hyb == Chem.rdchem.HybridizationType.SP:
        return _ANGLE_SP
    if hyb == Chem.rdchem.HybridizationType.SP3:
        return _ANGLE_SP3

    # Ring atoms: use ring size at this atom
    ri = mol.GetRingInfo()
    if ri.NumAtomRings(j) > 0:
        for ring in ri.AtomRings():
            if j in ring:
                size = len(ring)
                return _RING_ANGLE.get(size, _ANGLE_SP2)

    return _ANGLE_SP3


def _reference_angles_from_embed(mol: Chem.Mol, triplets: List[Tuple[int, int, int]]) -> np.ndarray:
    """Measure angles from a single embedded conformer."""
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    if AllChem.EmbedMolecule(mol_h, params) != 0:
        return np.array([_ANGLE_SP3] * len(triplets), dtype=np.float64)

    conf = mol_h.GetConformer()
    angles = []
    for i, j, k in triplets:
        pi = np.array(conf.GetAtomPosition(i))
        pj = np.array(conf.GetAtomPosition(j))
        pk = np.array(conf.GetAtomPosition(k))
        v1 = pi - pj
        v2 = pk - pj
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            angles.append(_ANGLE_SP3)
            continue
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angles.append(float(np.arccos(cos_a)))
    return np.array(angles, dtype=np.float64)


def _collect_torsions(
    mol: Chem.Mol, mmff_props, ff_mol: Chem.Mol | None = None
) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    """Collect MMFF periodic torsion parameters without selecting a target conformer."""
    quads: List[Tuple[int, int, int, int]] = []
    params_out: List[Tuple[float, float, float]] = []
    if mmff_props is None:
        return quads, np.empty((0, 3), dtype=np.float64)
    seen = set()
    for bond in mol.GetBonds():
        j, k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        left = [a.GetIdx() for a in mol.GetAtomWithIdx(j).GetNeighbors() if a.GetIdx() != k]
        right = [a.GetIdx() for a in mol.GetAtomWithIdx(k).GetNeighbors() if a.GetIdx() != j]
        for i in left:
            for l in right:
                key = min((i, j, k, l), (l, k, j, i))
                if key in seen:
                    continue
                ff = mmff_props.GetMMFFTorsionParams(ff_mol or mol, i, j, k, l)
                if ff is None:
                    continue
                seen.add(key)
                quads.append((i, j, k, l))
                params_out.append((float(ff[1]), float(ff[2]), float(ff[3])))
    return quads, np.asarray(params_out, dtype=np.float64).reshape(-1, 3)


def _embed_reference_coords(mol: Chem.Mol) -> np.ndarray | None:
    mol_h = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    if AllChem.EmbedMolecule(mol_h, params) != 0:
        return None
    conf = mol_h.GetConformer()
    return np.asarray([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])


def _collect_chirality(mol: Chem.Mol) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    """Use an isomeric-SMILES-consistent embed to define heavy-atom handedness signs."""
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    ref = _embed_reference_coords(mol)
    if ref is None:
        return [], np.empty(0, dtype=np.float64)
    centers: List[Tuple[int, int, int, int]] = []
    signs: List[float] = []
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            continue
        neighbors = sorted(n.GetIdx() for n in atom.GetNeighbors())
        if len(neighbors) < 3:
            continue
        a, b, c = neighbors[:3]
        center = atom.GetIdx()
        volume = np.dot(np.cross(ref[a] - ref[center], ref[b] - ref[center]), ref[c] - ref[center])
        if abs(volume) < 1e-8:
            continue
        centers.append((a, center, b, c))
        signs.append(1.0 if volume > 0 else -1.0)
    return centers, np.asarray(signs, dtype=np.float64)


def _collect_planar_groups(mol: Chem.Mol) -> List[Tuple[int, ...]]:
    groups: set[Tuple[int, ...]] = set()
    for ring in mol.GetRingInfo().AtomRings():
        # Five- and six-member aromatic rings are rigidly planar. Larger
        # non-benzenoid aromatic rings may pucker; enforce only local sp2
        # planarity for them below instead of an incorrect global plane.
        if len(ring) in (5, 6) and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            groups.add(tuple(sorted(ring)))
    for atom in mol.GetAtoms():
        if atom.GetHybridization() != Chem.rdchem.HybridizationType.SP2 and not atom.GetIsAromatic():
            continue
        group = tuple(sorted([atom.GetIdx(), *(n.GetIdx() for n in atom.GetNeighbors())]))
        if len(group) >= 4:
            groups.add(group)
    return sorted(groups, key=lambda x: (len(x), x))


def _collect_clashes(mol: Chem.Mol) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Non-bonded pairs beyond 1-2/1-3 topology with soft vdW lower bounds."""
    topo_dist = Chem.GetDistanceMatrix(mol)
    pt = Chem.GetPeriodicTable()
    pairs: List[Tuple[int, int]] = []
    limits: List[float] = []
    for i in range(mol.GetNumAtoms()):
        for j in range(i + 1, mol.GetNumAtoms()):
            if topo_dist[i, j] <= 2:
                continue
            ri = pt.GetRvdw(mol.GetAtomWithIdx(i).GetAtomicNum())
            rj = pt.GetRvdw(mol.GetAtomWithIdx(j).GetAtomicNum())
            pairs.append((i, j))
            limits.append(0.62 * float(ri + rj))
    return pairs, np.asarray(limits, dtype=np.float64)


def build_topology(
    smiles: str,
    use_embed_reference_angles: bool = True,
    mol: Chem.Mol | None = None,
) -> MoleculeTopology:
    """Build geometry targets in the supplied molecule's atom order.

    Without ``mol``, parse SMILES without adding hydrogens. Pass an explicit-H
    molecule to include those hydrogens in the optimization.
    """
    mol = Chem.Mol(mol) if mol is not None else mol_from_smiles(smiles)
    ff_mol = Chem.Mol(mol) if any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()) else Chem.AddHs(Chem.Mol(mol))
    mmff_props = AllChem.MMFFGetMoleculeProperties(ff_mol, mmffVariant="MMFF94s")
    bond_edges, target_bond_lengths = _collect_bonds(mol, mmff_props, ff_mol)
    angle_triplets, angle_targets_list = _collect_angles(mol, mmff_props, ff_mol)
    # use_embed_reference_angles is retained for compatibility; targets use force-field typing.
    target_angles = np.array(angle_targets_list, dtype=np.float64)
    torsion_quads, torsion_params = _collect_torsions(mol, mmff_props, ff_mol)
    chiral_centers, chiral_signs = _collect_chirality(mol)
    planar_groups = _collect_planar_groups(mol)
    nonbonded_pairs, clash_distances = _collect_clashes(mol)

    return MoleculeTopology(
        mol=mol,
        smiles=smiles,
        bond_edges=bond_edges,
        angle_triplets=angle_triplets,
        target_bond_lengths=target_bond_lengths,
        target_angles=target_angles,
        torsion_quads=torsion_quads,
        torsion_params=torsion_params,
        chiral_centers=chiral_centers,
        chiral_signs=chiral_signs,
        planar_groups=planar_groups,
        nonbonded_pairs=nonbonded_pairs,
        clash_distances=clash_distances,
    )


def get_coords_from_mol(mol: Chem.Mol, conf_id: int = 0) -> np.ndarray:
    """Extract (N, 3) coordinates from mol conformer (heavy atoms only)."""
    conf = mol.GetConformer(conf_id)
    n = mol.GetNumAtoms()
    return np.array([list(conf.GetAtomPosition(i)) for i in range(n)], dtype=np.float64)
