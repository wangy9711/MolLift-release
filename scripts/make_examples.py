"""Generate synthetic RDKit molecules and projection annotations."""

from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from mollift.io import write_json


def create(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    for name, smiles, explicit in [
        ("small_molecule", "CCOC(=O)c1ccccc1", True),
        ("peptide", "CC(=O)NCC(=O)N[C@@H](C)C(=O)NCC(=O)N[C@@H](C)C(=O)NCC(=O)NC", False),
    ]:
        heavy = Chem.MolFromSmiles(smiles)
        full = Chem.AddHs(heavy)
        if AllChem.EmbedMolecule(full, randomSeed=42) != 0:
            raise RuntimeError("Could not create example")
        AllChem.MMFFOptimizeMolecule(full)
        mol = full if explicit else Chem.RemoveHs(full)
        xyz = np.asarray(mol.GetConformer().GetPositions())
        xyz -= xyz.mean(0)
        mask = np.random.default_rng(42).random(len(xyz)) < 0.8
        mask[:2] = True
        xy = xyz[:, :2].copy()
        xy[~mask] = 0
        write_json(root / f"{name}.json", {
            "smiles": smiles, "units": "angstrom", "centering": "provided",
            "atom_representation": "explicit_h" if explicit else "heavy",
            "atomic_numbers": [a.GetAtomicNum() for a in mol.GetAtoms()],
            "coords_2d": xy, "observed_mask": mask,
        })
        if not explicit:
            pixels = xyz[:, :2] * [40, -40] + [400, 300]
            write_json(root / "peptide_annotation.json", {
                "format_version": "1.0", "smiles": smiles,
                "image": {"path": "peptide_projection.png", "width": 800, "height": 600},
                "num_atoms": len(xyz), "annotated_mask": mask,
                "annotations": [{"atom_index": i, "symbol": a.GetSymbol(), "pixel": pixels[i]}
                                for i, a in enumerate(mol.GetAtoms()) if mask[i]],
                "skipped_atoms": [], "metadata": {"source": "RDKit synthetic demo, not a paper case"},
            })
            from PIL import Image, ImageDraw
            img = Image.new("RGB", (800, 600), "white")
            draw = ImageDraw.Draw(img)
            for bond in mol.GetBonds():
                left, right = pixels[bond.GetBeginAtomIdx()], pixels[bond.GetEndAtomIdx()]
                draw.line([tuple(left), tuple(right)], fill="#777777", width=3)
            for i, (a, point) in enumerate(zip(mol.GetAtoms(), pixels)):
                x, y = point
                color = {"O": "#d62728", "N": "#1f77b4"}.get(a.GetSymbol(), "#444444")
                draw.ellipse((x-5, y-5, x+5, y+5), fill=color)
                draw.text((x+6, y+4), f"{i}:{a.GetSymbol()}", fill="black")
            img.save(root / "peptide_projection.png")


if __name__ == "__main__":
    create(Path("examples"))
