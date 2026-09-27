"""Tests for partial projection masks."""

import numpy as np
import torch

from mollift_generator.data.proj_mask import (
    pair_mask_from_proj_mask,
    proj_mask_from_indices,
    sample_proj_mask_deterministic,
    sample_proj_mask_random,
    sample_proj_mask_random_choices,
)


def test_ratio_one_all_observed():
    node_mask = torch.tensor([[True, True, True, False]])
    mask = sample_proj_mask_random(node_mask, 1.0)
    assert mask.equal(node_mask)


def test_ratio_zero_none_observed():
    node_mask = torch.tensor([[True, True, True, False]])
    mask = sample_proj_mask_random(node_mask, 0.0)
    assert not mask.any()


def test_ratio_partial_count():
    node_mask = torch.tensor([[True] * 10 + [False] * 5], dtype=torch.bool)
    mask = sample_proj_mask_random(node_mask, 0.5)
    assert int(mask.sum()) == 5


def test_mixed_ratio_samples_choice_independently_per_graph():
    node_mask = torch.ones((32, 10), dtype=torch.bool)
    torch.manual_seed(7)
    mask = sample_proj_mask_random_choices(node_mask, [0.3, 0.5, 0.8, 1.0])
    counts = set(mask.sum(dim=1).tolist())
    assert counts.issubset({3, 5, 8, 10})
    assert len(counts) > 1


def test_mixed_ratio_rejects_invalid_choices():
    node_mask = torch.ones((1, 4), dtype=torch.bool)
    for choices in ([], [0.5, 1.1]):
        try:
            sample_proj_mask_random_choices(node_mask, choices)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid choices to fail: {choices}")


def test_deterministic_reproducible():
    node_mask = torch.tensor([[True, True, True, True, False]])
    m1 = sample_proj_mask_deterministic(node_mask, 0.5, [7], seed=42)
    m2 = sample_proj_mask_deterministic(node_mask, 0.5, [7], seed=42)
    assert m1.equal(m2)


def test_proj_mask_from_indices():
    mask = proj_mask_from_indices(5, [0, 2, 4])
    assert mask.tolist() == [True, False, True, False, True]


def test_pair_mask_requires_both_observed():
    proj_mask = torch.tensor([[True, True, False]])
    pair_mask = pair_mask_from_proj_mask(proj_mask)
    assert pair_mask[0, 0, 0]
    assert pair_mask[0, 0, 1]
    assert not pair_mask[0, 0, 2]
    assert not pair_mask[0, 2, 2]
