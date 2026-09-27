"""Inference tests against the two bundled pretrained checkpoints."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from mollift.io import load_input, sha256_file
from mollift.cli import registry
from mollift.predict import generate
from mollift_generator.models.checkpoint import load_model_from_checkpoint


@pytest.mark.parametrize("name,example", [("mollift-drugs", "small_molecule"), ("mollift-peptide", "peptide")])
def test_real_checkpoint_and_candidate_chunking(name, example):
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    entry = registry()[name]
    path = Path(os.environ.get("MOLLIFT_CHECKPOINT_DIR", root / "checkpoints")) / entry["filename"]
    if not path.exists():
        pytest.skip("Checkpoint missing; restore it from the repository or set MOLLIFT_CHECKPOINT_DIR")
    assert sha256_file(path) == entry["sha256"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model_from_checkpoint(path, device)
    molecule = load_input(root / "examples" / f"{example}.json")
    one, _ = generate(molecule, model, cfg, num_samples=2, batch_size=1, device=device)
    two, _ = generate(molecule, model, cfg, num_samples=2, batch_size=2, device=device)
    assert np.isfinite(two).all()
    assert np.allclose(one, two, atol=3e-5, rtol=1e-5)
