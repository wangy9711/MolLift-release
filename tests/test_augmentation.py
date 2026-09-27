"""Tests for rotation augmentation."""

import torch

from mollift_generator.training.augmentation import apply_rotation_augmentation


def test_rotation_augmentation_xy_matches_rotated_x3d():
    batch = {
        "x3d": torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
        "pos_2d": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        "node_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    out = apply_rotation_augmentation(batch)
    d3 = (out["x3d"] ** 2).sum(dim=-1)
    assert torch.allclose(d3, torch.tensor([[1.0, 1.0]]), atol=1e-5)
    assert torch.allclose(out["pos_2d"], out["x3d"][..., :2], atol=1e-5)


def test_rotation_augmentation_projection_consistent_nonplanar():
    """Projection after rotation must include the original depth contribution."""
    torch.manual_seed(0)
    n = 10
    x3d = torch.randn(1, n, 3)
    x3d = x3d - x3d.mean(dim=1, keepdim=True)  # centered
    batch = {
        "x3d": x3d,
        "pos_2d": x3d[..., :2].clone(),  # synthetic projection = xy slice
        "node_mask": torch.ones(1, n, dtype=torch.bool),
    }
    assert x3d[..., 2].abs().max() > 0.1  # ensure genuinely non-planar
    out = apply_rotation_augmentation(batch)
    assert torch.allclose(out["pos_2d"], out["x3d"][..., :2], atol=1e-5)
    assert not torch.allclose(out["x3d"], x3d, atol=1e-3)
