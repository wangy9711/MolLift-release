# Data preparation and training

Commands assume the repository root is the working directory. Edit the paths in
`configs/*.yaml` for your data location. Paths are relative to the working directory,
not the configuration file. Training settings reproduce the released Mixed designs;
memory-related batch changes also change the effective training protocol.

## GEOM-Drugs

Obtain the trusted GEOM RDKit pickle release from its original distribution:
[GEOM](https://github.com/learningmatter-mit/geom). Obtain GeoMol's official
`data/DRUGS/` split files from [GeoMol](https://github.com/PattanaikL/GeoMol).
These external datasets retain their own distribution terms. They are not included
in this repository. Required GeoMol files are `splits/split0.npy`,
`smiles_splits/split0.npy` and `test_smiles_corrected.csv`.

```bash
python -m mollift_generator.scripts.precompute_geom_cache \
  --data-root /path/to/rdkit_folder/drugs --cache-root data/geom_drugs \
  --dataset-name drugs --max-atoms 181 --max-hop 8 --workers 8

python -m mollift_generator.scripts.import_geomol_splits \
  --geomol-data /path/to/GeoMol/data --drugs-cache data/geom_drugs \
  --output-root data/splits
```

Cache construction is resumable. It creates an initial hash split for processing;
**importing GeoMol split0 is required before paper-protocol training**. The public
training entry checks `index.sqlite` split metadata. Molecules, not conformers,
define split membership. `test` and `test_pool` are never used for training.
Keep the mapping report, preprocessing summary and failed-record logs with your
run. The published Drugs cache had 242,077 train and 30,338 valid molecules;
source versions, RDKit differences and filtering can change a fresh build's counts.

```bash
CUDA_VISIBLE_DEVICES=0 python -m mollift train --config configs/drugs_mixed.yaml
```

The default is 200 epochs, batch size 32, AdamW peak learning rate 3e-4,
weight decay 0.01, warmup/cosine schedule and EMA decay 0.999. Each molecule samples
one conformer per epoch and a projection observation ratio uniformly from
`[0.3, 0.5, 0.8, 1.0]`. Best checkpoint selection averages the four corresponding
validation losses. The atom/bond vocabulary remains 72/5. An original worst-case
batch of 32 at 181 atoms used about 22.2 GiB; measure your environment before a
long run.

For multiple GPUs on one machine:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m mollift train --config configs/drugs_mixed.yaml
```

Batch size is per GPU. Changing world size changes effective batch size and the
number of optimizer steps. The distributed path uses NCCL; no machine-specific
NCCL overrides are forced by this release.

## Peptide adaptation

The original PDB-derived package is **not publicly distributed**. See
[the construction overview](pdb_data.md) for background. Training requires a
private package compatible with the included data loader; these documents do not
provide a package-format specification.

```bash
# The Drugs EMA checkpoint is included at checkpoints/mollift-drugs.pt.
# Set data.pdb_root and data.drugs_cache_root in the config.
CUDA_VISIBLE_DEVICES=0 python -m mollift train --config configs/peptide_mixed.yaml
```

This trainer uses one GPU, length curriculum, atom-count buckets and graph-count
gradient accumulation. It runs 30 epochs from the matched Drugs Mixed EMA model,
with 75% peptide / 25% Drugs replay and a peak learning rate of 3e-5. The primary
best checkpoint must keep four-ratio Drugs validation within 5% of a fixed
pre-transfer baseline. The `best_pdb` snapshot ignores retention and is diagnostic;
use the primary `best` for deployment. The released model came from epoch 20.

## Logs and continuing a run

Both configurations enable `train.resume`. Rerunning the same command resumes
from the corresponding latest training checkpoint when present. The public EMA-only
model assets initialize adaptation; they cannot restore optimizer state. 
