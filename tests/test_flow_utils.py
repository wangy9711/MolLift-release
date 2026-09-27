"""Tests for flow_utils (dit_mc-aligned options)."""

import torch

from mollift_generator.training.flow_utils import (
    build_prior_graph,
    kabsch_align_batch,
    kabsch_align_batch_gpu,
    kabsch_align_batch_numpy,
    sample_flow_batch,
    sample_harmonic_prior,
)
from mollift_generator.utils.tensor_utils import center_coords


def test_kabsch_align_batch_identity():
    node_mask = torch.tensor([[True, True, True]])
    target = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    aligned = kabsch_align_batch(target, target, node_mask)
    err = ((aligned - target) ** 2).sum().sqrt()
    assert err < 1e-5


def test_kabsch_gpu_half_precision():
    if not torch.cuda.is_available():
        return
    node_mask = torch.ones(2, 5, dtype=torch.bool, device="cuda")
    source = torch.randn(2, 5, 3, device="cuda", dtype=torch.float16)
    target = torch.randn(2, 5, 3, device="cuda", dtype=torch.float16)
    with torch.autocast("cuda", dtype=torch.float16):
        aligned = kabsch_align_batch_gpu(source, target, node_mask)
    assert aligned.dtype == torch.float16
    assert torch.isfinite(aligned).all()


def test_sample_flow_batch_under_autocast():
    if not torch.cuda.is_available():
        return
    b, n = 2, 5
    x1 = torch.randn(b, n, 3, device="cuda")
    node_mask = torch.ones(b, n, dtype=torch.bool, device="cuda")
    bond = torch.zeros(b, n, n, 6, device="cuda")
    bond[:, 0, 1, 0] = 1
    bond[:, 1, 0, 0] = 1
    bond[:, 1, 2, 0] = 1
    bond[:, 2, 1, 0] = 1
    with torch.autocast("cuda", dtype=torch.float16):
        x_t, t, x1_out = sample_flow_batch(
            x1,
            node_mask,
            sigma=0.08,
            prior_type="harmonic",
            align_x0=True,
            bond_features_dense=bond,
        )
    assert torch.isfinite(x_t).all()
    assert x_t.dtype == torch.float32


def test_kabsch_gpu_matches_numpy():
    torch.manual_seed(0)
    b, n = 4, 8
    source = torch.randn(b, n, 3)
    target = torch.randn(b, n, 3)
    node_mask = torch.ones(b, n, dtype=torch.bool)
    node_mask[:, -2:] = False

    aligned_np = kabsch_align_batch_numpy(source, target, node_mask)
    if torch.cuda.is_available():
        aligned_gpu = kabsch_align_batch_gpu(
            source.cuda(), target.cuda(), node_mask.cuda()
        ).cpu()
        err = ((aligned_gpu - aligned_np) ** 2).sum().sqrt()
        assert err < 1e-4


def test_sample_flow_batch_harmonic_prior():
    b, n = 2, 5
    x1 = torch.randn(b, n, 3)
    node_mask = torch.ones(b, n, dtype=torch.bool)
    node_mask[0, 4] = False
    bond = torch.zeros(b, n, n, 6)
    bond[0, 0, 1, 0] = 1
    bond[0, 1, 0, 0] = 1
    bond[1, 0, 1, 0] = 1
    bond[1, 1, 0, 0] = 1
    x_t, t, x1_out = sample_flow_batch(
        x1,
        node_mask,
        sigma=0.5,
        prior_type="harmonic",
        align_x0=True,
        bond_features_dense=bond,
    )
    assert x_t.shape == x1.shape
    assert t.shape == (b,)
    assert torch.all(t >= 0.0) and torch.all(t <= 1.0)
    assert torch.allclose(x1_out, center_coords(x1, node_mask))
    assert torch.isfinite(x_t).all()


def test_harmonic_prior_is_centered():
    bond = torch.zeros(1, 4, 4, 6)
    bond[0, 0, 1, 0] = 1
    bond[0, 1, 0, 0] = 1
    bond[0, 1, 2, 0] = 1
    bond[0, 2, 1, 0] = 1
    node_mask = torch.ones(1, 4, dtype=torch.bool)
    x0 = sample_harmonic_prior(bond, node_mask)
    mean = (x0 * node_mask.unsqueeze(-1)).sum(dim=1) / node_mask.sum(dim=1, keepdim=True).clamp_min(1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-3)


def test_harmonic_prior_zero_mode_not_amplified():
    bond = torch.zeros(1, 4, 4, 6)
    bond[0, 0, 1, 0] = 1
    bond[0, 1, 0, 0] = 1
    bond[0, 1, 2, 0] = 1
    bond[0, 2, 1, 0] = 1
    node_mask = torch.ones(1, 4, dtype=torch.bool)
    prior = build_prior_graph(bond, node_mask)
    assert prior.node_attr[0, 0].item() == 0.0


def test_harmonic_prior_bounded_magnitude():
    bond = torch.zeros(1, 8, 8, 6)
    for i in range(7):
        bond[0, i, i + 1, 0] = 1
        bond[0, i + 1, i, 0] = 1
    node_mask = torch.ones(1, 8, dtype=torch.bool)
    x0 = sample_harmonic_prior(bond, node_mask)
    assert torch.isfinite(x0).all()
    assert x0.abs().max().item() < 50.0


def test_harmonic_prior_matches_einsum_formula():
    torch.manual_seed(0)
    m = 5
    P = torch.linalg.qr(torch.randn(m, m))[0]
    node_a = torch.tensor([0.0, 1.2, 0.8, 0.5, 0.3])
    noise = torch.randn(m, 3)
    scaled = node_a.unsqueeze(-1) * noise
    expected = torch.einsum("ij,jd->id", P, scaled)

    bond = torch.zeros(1, m, m, 6)
    for i in range(m - 1):
        bond[0, i, i + 1, 0] = 1
        bond[0, i + 1, i, 0] = 1
    node_mask = torch.ones(1, m, dtype=torch.bool)
    prior = build_prior_graph(bond, node_mask)
    prior.evecs[0, :m, :m] = P
    prior.node_attr[0, :m] = node_a

    from mollift_generator.training.flow_utils import _harmonic_prior_single

    sample = _harmonic_prior_single(prior, 0, m, P.device, P.dtype, noise=noise)
    assert torch.allclose(sample, expected, atol=1e-5)
