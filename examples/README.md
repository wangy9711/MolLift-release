# Synthetic examples

These files were generated using RDKit ETKDG/MMFF with seed 42. They demonstrate
the public interface; they are not paper benchmark cases or evidence of accuracy.
No private peptide data or literature images are included.

- `small_molecule.json`: explicit-H ethyl benzoate with partial calibrated projection.
- `peptide.json`: capped five-residue peptide, heavy-only, with partial calibrated projection.
- `peptide_annotation.json`: visible pixels for the same synthetic peptide.
- `peptide_projection.png`: a synthetic projection image with atom indices.

Use 40 pixels/angstrom for this example annotation. The pixel path centers the
observed subset, whereas the calibrated JSON preserves the original whole-molecule
centering, so the two inputs intentionally illustrate different centering conventions.
The prediction code never reads the 3D structure used to create the example.

Regenerate with `PYTHONPATH=. python scripts/make_examples.py` (requires Pillow).
