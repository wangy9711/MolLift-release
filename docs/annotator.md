# Desktop atom annotation

Install the optional desktop dependencies:

```bash
python -m pip install -e '.[annotator]'
mollift-annotator --smiles 'CCO' --image /path/to/projection.png --output annotation.json
```

Use the exact isomeric SMILES for the molecule. The right panel shows its molecular
graph and atom indices; the left panel shows the image. Click a visible point and
its corresponding graph atom, or select an atom first and then click its image
location. Save the JSON; Load resumes a previous annotation. Invisible atoms may
remain unannotated or be marked skipped. Do not invent pixel coordinates for them.

Mouse wheel zooms; middle-button drag or Space+left drag pans. Toolbar controls
provide zoom, fit and 100% for each panel. The tool exports `format_version`,
`smiles`, image metadata, `num_atoms`, `annotated_mask`, per-atom `annotations`
and `skipped_atoms`. Each observation records `atom_index`, `symbol` and `pixel`.

```bash
python -m mollift predict --model mollift-peptide \
  --annotation annotation.json --pipeline GSR --output outputs/annotated
```

Provide `--pixels-per-angstrom` when you know the scale; otherwise the runner
estimates it from visible bonded atoms. See [calibration limitations](inference.md).
The image is used for human annotation only; the inference command consumes JSON.
Do not treat conventional 2D structural formulas as exact orthographic 3D views.

The included `examples/peptide_projection.png` and `peptide_annotation.json`
were generated with RDKit. They can be used to test JSON loading and the GUI
without redistributing literature illustrations. Their SMILES is stored in the JSON.
The original graph/annotation functions are retained; GUI launch requires a display
and PySide6.
