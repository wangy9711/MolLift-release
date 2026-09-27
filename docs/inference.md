# Inference inputs and outputs

## Calibrated coordinates

The JSON schema requires a SMILES, its modeled atom numbers in order, coordinates
in angstroms and an observation mask. Atom indices are zero-based. For `heavy`,
use RDKit `MolFromSmiles(smiles)` order. For `explicit_h`, use
`Chem.AddHs(Chem.MolFromSmiles(smiles))` order (hydrogens are appended). The exact
SMILES spelling matters for row order; do not canonicalize it without remapping.

```json
{
  "smiles": "CCO",
  "units": "angstrom",
  "atom_representation": "heavy",
  "atomic_numbers": [6, 6, 8],
  "coords_2d": [[-0.8, 0.0], [0.5, 0.3], [null, null]],
  "observed_mask": [true, true, false],
  "centering": "provided"
}
```

The above coordinates illustrate the schema, not an experimental reference.
Hidden rows are always zeroed before featurization. At least one atom must be
observed; disconnected mixtures and non-finite observed coordinates are rejected.
This interface is for conditioned reconstruction, not an unconditioned benchmark.

`centering: provided` preserves a supplied calibrated origin. `centering: observed`
subtracts the visible-atom centroid. Synthetic training centers the entire molecule
before hiding coordinates, whereas image annotation uses the observed centroid;
these are different protocols. Partial-image centering can introduce a distribution
shift. No target structure is used to determine scale or origin at inference.

## Pixel annotations

```bash
python -m mollift predict --model mollift-peptide \
  --annotation annotation.json --pipeline GSR --output outputs/my_peptide
```

The annotation schema is documented by the annotator and includes SMILES atom indices.
Pixel y is flipped and visible coordinates are centered. With
`--pixels-per-angstrom VALUE`, the supplied scale is used. Otherwise, the median
of observed bonded-pair pixel distances divided by topology-derived target bond
lengths estimates the scale. Auto-calibration needs at least one observed bond;
projected bond lengths may be foreshortened, so this estimate is approximate.
The raw image and a crystal/SDF reference are not required to run an annotation JSON.

## Processing settings

G uses the released EMA model, graph-harmonic initial noise and 50 Euler steps.
The default is K=10. Candidate i uses seed+i; candidate chunking changes only batch
size (floating-point differences may remain). This seed convention is public and
deterministic but is not the historical query-ID hashing used in benchmark queues.

Current method applies a global z reflection when all checked, four-neighbor declared
tetrahedral centers are inverted. It does not repair local stereochemical errors;
implicit-H centers without four modeled neighbors are not handled by that check.
The run records the correction decisions. Supplied stereochemistry and local
geometry still require scientific assessment, especially on out-of-domain inputs.

The Solver (S) uses Adam, learning rate 0.035, 600 steps,
first 30% at 0.1 times the reprojection weight, full-objective best-state tracking,
stage-reset patience and at least 30 full-weight steps. Weights are
projection/bond/angle/torsion/chirality/planarity/clash =
20/10/5/0.15/5/10/2. No extra mirror candidates are added. `--solver-steps` is
available for diagnostics; changing it changes the protocol.

R uses Open Babel MMFF94s with UFF fallback: 50 hydrogen-relaxation steps, up to
five coordinate-update steps, maximum heavy-atom RMS displacement 0.05 Å, maximum
modeled-atom displacement 0.15 Å and maximum component projection-RMSE increase
0.03 Å. Clash, specified stereo and energy acceptance checks are also enforced.
Hydrogens added internally for heavy-only force-field calculations are not returned.
All K candidates are processed; no top-M prefilter is used.

The G output is the first sample. GS selects the smallest final Solver objective.
GR/GSR first keep accepted candidates whose observed **component** projection RMSE
is within **0.10 Å of the eligible minimum**, then select minimum energy within
the highest-priority available force-field group (MMFF94s, MMFF94, UFF). This is
a relative window, not an absolute error bound. No reference coordinates or RMSD
enter this function. No valid candidate means an explicit failed run, not a silent
fallback or extra sampling.

## Output contract

| File | Content |
|---|---|
| `selected.sdf` | Reference-free selected structure in the observation frame |
| `generator.sdf` | All finite G candidates with original candidate indices |
| `solver.sdf` | GS candidates, when S is enabled |
| `refined.sdf` | Accepted refined candidates, when R is enabled |
| `candidates.npz` | Full arrays for each executed stage; failed candidate slots remain NaN |
| `input.json` | Normalized, masked observation and authoritative atom order |
| `model_config.json` | Model settings loaded from the checkpoint |
| `run.json` | Hash, seeds, versions, settings, timings, errors, energy and selection diagnostics |
| `solver_diagnostics.json` | Solver trajectories and returned-state information, when S is enabled |

No GT-aligned output is written. `run.json` uses null for unavailable/non-finite
metrics; NPZ preserves failed slots. Component projection RMSE averages squared
error over both xy components and visible atoms. Vector RMSE is sqrt(2) times this
value; do not interchange the two when reproducing the selection tolerance.

Stage arrays in `candidates.npz` are named `G`, `GS`, `GR` or `GSR`, according to
the selected pipeline. `run.json` records the Refiner duration as `timing_seconds.R`.

The output directory must not already exist. This prevents accidental overwriting
of completed scientific results. After inspecting a failed run, retry in a new
directory. To select a GPU, use `CUDA_VISIBLE_DEVICES=1` and `--device cuda:0`;
the CUDA index is relative to the visible devices. Several independent commands
may run concurrently, each with its own output directory.
