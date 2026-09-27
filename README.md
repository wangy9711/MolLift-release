# MolLift

Recover a molecular conformer from a molecular graph and **atom-aligned planar
observations**. MolLift combines a projection-conditioned flow Generator (G),
an optional geometric Solver (S), and a constrained force-field Refiner (R).

The input is an isomeric SMILES plus observed atom positions, or a JSON annotation
exported by the included desktop annotator. MolLift does not recognize a molecular
graph from an image. Its reconstruction assumption is an approximately orthographic
view of a 3D conformation; arbitrary 2D chemical drawing layouts are not equivalent
to such observations.

## Install

Use Python 3.10 or newer. Install a PyTorch build appropriate for your machine,
then run from this directory:

```bash
python -m pip install -e '.[refiner,train]'
python -m mollift --help
```

The base install supports Generator and Solver. `refiner` adds Open Babel;
`train` adds peptide data readers. The separate `annotator` extra adds the Qt GUI:

```bash
python -m pip install -e '.[annotator]'
```

Linux is the validated inference/training platform. CUDA is recommended for G;
CPU inference is also supported. The Solver defaults to CPU.
See [installation](docs/installation.md).

## Pretrained models

Two EMA-only checkpoint files are included in this Git repository under
`checkpoints/`. No separate model download or training data are needed for inference.
If you move the checkpoints, set `MOLLIFT_CHECKPOINT_DIR` to their directory.

| Model name | File | Intended use |
|---|---|---|
| `mollift-drugs` | `mollift-drugs.pt` | Small and drug-like molecules; GEOM-Drugs Mixed |
| `mollift-peptide` | `mollift-peptide.pt` | Peptides; PDB-adapted Mixed |

```bash
python -m mollift models --check
```

Both models were trained with observation ratios sampled from
`[0.3, 0.5, 0.8, 1.0]`. They are practical choices when observation completeness is
unknown. Full-observation specialist checkpoints are not included.
See [model descriptions](docs/models.md).

## Inference

Run the small synthetic example:

```bash
python -m mollift predict --model mollift-drugs \
  --input examples/small_molecule.json --pipeline GR \
  --device cuda:0 --output outputs/small_molecule
```

For a peptide annotation, including geometric repair:

```bash
python -m mollift predict --model mollift-peptide \
  --annotation examples/peptide_annotation.json --pixels-per-angstrom 40 \
  --pipeline GSR --device cuda:0 --batch-size 2 --output outputs/peptide
```

Use `--device cpu` without CUDA. Outputs must use a new directory. Default sampling
is 10 candidates with 50 Euler steps; `--batch-size` controls memory use independently
of the candidate count.

| Pipeline | Processing | Reference-free output selection |
|---|---|---|
| `G` | Generator | First sample |
| `GR` (default) | Generator → Refiner | Projection window, then comparable force-field energy |
| `GS` | Generator → Solver | Lowest final Solver objective |
| `GSR` | Generator → Solver → Refiner | Projection window, then comparable force-field energy |

`GR` is the default supported by synthetic partial-observation experiments.
`GSR` provides optional repair for difficult image-derived observations; it does not
improve every molecule. No ground-truth structure or reference RMSD is used to select
an output. See [input formats, outputs and pipeline settings](docs/inference.md).

Each successful run writes `selected.sdf`, all finite candidates as SDF, an NPZ
containing the full candidate axes, normalized input, and `run.json` with seeds,
checkpoint hash, timings, failures, projection errors and selection diagnostics.

## Train

After preparing the public GEOM-Drugs cache and importing GeoMol split0:

```bash
CUDA_VISIBLE_DEVICES=0 python -m mollift train --config configs/drugs_mixed.yaml
```

Peptide transfer uses the Drugs EMA checkpoint, a compatible private peptide
package and GEOM-Drugs replay:

```bash
CUDA_VISIBLE_DEVICES=0 python -m mollift train --config configs/peptide_mixed.yaml
```

The PDB-derived training dataset **is not distributed**. The transfer implementation
and configuration are included, but reproducing that training requires supplying
a compatible peptide package. See [PDB data construction](docs/pdb_data.md) for
background. Pretrained peptide inference works without it.
See [data preparation and training](docs/training.md).

## Annotate an image

```bash
mollift-annotator \
  --smiles 'CCO' --image /path/to/projection.png --output annotation.json
```

The desktop tool binds visible image points to SMILES atom indices and supports
partial annotation. Exported JSON is accepted directly by `mollift predict`.
See [annotator instructions](docs/annotator.md). The images and inputs in `examples/`
are synthetic RDKit demonstrations, not the paper's literature images or PDB data.


## License and acknowledgements

Code is distributed under the [MIT License](LICENSE). MolLift builds on DiTMC's
graph-conditioned conformer-generation framework. Please retain the upstream
attribution. External
datasets and installed dependencies retain their own licenses.
