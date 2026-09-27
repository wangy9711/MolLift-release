# Pretrained checkpoints

`mollift-drugs.pt` and `mollift-peptide.pt` are included in this Git repository
alongside their identity JSON sidecars. No separate download is required.
If you move the files, set `MOLLIFT_CHECKPOINT_DIR` to their directory.
Run `python -m mollift models --check` to verify them.

Do not rename Full, unconstrained PDB-only, latest or QM9 checkpoints to these
filenames. Named-model inference checks the exact public-file hashes in
`mollift/models.json`. See [model documentation](../docs/models.md).
