import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from mollift_solver import SolveConfig, solve_3d
from mollift_solver.pipeline import MoleculeInput


def test_solver_preserves_authoritative_explicit_hydrogen_atom_set():
    smiles = "CCO"
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = 17
    assert AllChem.EmbedMolecule(mol, params) == 0
    coords = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    molecule = MoleculeInput(smiles, coords[:, :2], atomic_numbers)
    molecule.validate()

    result = solve_3d(
        smiles,
        coords[:, :2],
        coords_3d_init=coords,
        coords_2d_mask=np.ones(len(coords), dtype=bool),
        config=SolveConfig(steps=50, include_mirror_starts=False),
        mol=molecule.to_rdkit_mol(),
    )
    assert result.coords_3d.shape == coords.shape
    assert result.mol.GetNumAtoms() == mol.GetNumAtoms()
    assert [atom.GetAtomicNum() for atom in result.mol.GetAtoms()] == atomic_numbers
