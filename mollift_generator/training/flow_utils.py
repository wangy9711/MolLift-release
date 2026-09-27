"""Flow-matching utilities aligned with dit_mc options."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mollift_generator.training.metrics import kabsch_align
from mollift_generator.utils.tensor_utils import center_coords

_EIG_THRESHOLD = 1e-4


@dataclass
class PriorGraph:
    """Dense prior graph features for harmonic sampling (dit_mc HarmonicPrior)."""

    node_attr: torch.Tensor  # [B, N]  1/sqrt(eigenvalue), zero modes zeroed
    evecs: torch.Tensor  # [B, N, N]  eigenvector matrix P


def _count_connected_components(adj: torch.Tensor) -> int:
    """Count connected components in an undirected adjacency matrix [m, m]."""
    n = adj.shape[0]
    visited = torch.zeros(n, dtype=torch.bool, device=adj.device)
    count = 0
    for start in range(n):
        if visited[start]:
            continue
        count += 1
        stack = [start]
        visited[start] = True
        while stack:
            u = stack.pop()
            neighbors = (adj[u] > 0).nonzero(as_tuple=False).flatten()
            for v in neighbors.tolist():
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
    return count


def _mode_scaling_from_laplacian(lap: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    """DiTMC-style 1/sqrt(eigenvalue) with zero modes zeroed."""
    evals, _ = torch.linalg.eigh(lap)
    inv_sqrt = torch.where(
        evals > 0,
        1.0 / torch.sqrt(evals.clamp_min(1e-12)),
        torch.zeros_like(evals),
    )
    total = evals.sum().clamp_min(1e-12)
    ratio = evals / total
    inv_sqrt = torch.where(ratio < _EIG_THRESHOLD, torch.zeros_like(inv_sqrt), inv_sqrt)
    n_comp = _count_connected_components(adj)
    inv_sqrt[:n_comp] = 0
    return inv_sqrt


def build_prior_graph(
    bond_features_dense: torch.Tensor,
    node_mask: torch.Tensor,
) -> PriorGraph:
    """Build dit_mc-style prior graph from bond adjacency."""
    with torch.no_grad():
        b, n, _, _ = bond_features_dense.shape
        device = bond_features_dense.device
        dtype = bond_features_dense.dtype
        bond_adj = (bond_features_dense[..., :4].sum(dim=-1) > 0).to(dtype)

        node_attr = torch.zeros(b, n, device=device, dtype=dtype)
        evecs = torch.zeros(b, n, n, device=device, dtype=dtype)

        counts = node_mask.sum(dim=1).tolist()
        for g, m in enumerate(counts):
            m = int(m)
            if m <= 0:
                continue
            adj = bond_adj[g, :m, :m]
            lap = torch.diag(adj.sum(dim=-1)) - adj
            _, P = torch.linalg.eigh(lap)
            node_attr[g, :m] = _mode_scaling_from_laplacian(lap, adj)
            evecs[g, :m, :m] = P

    return PriorGraph(node_attr=node_attr, evecs=evecs)


def _harmonic_prior_single(
    prior: PriorGraph,
    graph_idx: int,
    n_atoms: int,
    device: torch.device,
    dtype: torch.dtype,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """DiTMC HarmonicPrior: sample[j] = sum_i P[i,j] * node_attr[i] * noise[i]."""
    if noise is None:
        noise = torch.randn(n_atoms, 3, device=device, dtype=dtype)
    node_a = prior.node_attr[graph_idx, :n_atoms]
    P = prior.evecs[graph_idx, :n_atoms, :n_atoms]
    scaled = node_a.unsqueeze(-1) * noise
    sample = torch.einsum("ij,jd->id", P, scaled)
    return torch.nan_to_num(sample)


def sample_harmonic_prior(
    bond_features_dense: torch.Tensor,
    node_mask: torch.Tensor,
    prior: PriorGraph | None = None,
) -> torch.Tensor:
    """Batch harmonic prior via message passing on prior graph."""
    if prior is None:
        prior = build_prior_graph(bond_features_dense, node_mask)
    b, n, _, _ = bond_features_dense.shape
    device = bond_features_dense.device
    dtype = bond_features_dense.dtype

    noise = torch.randn(b, n, 3, device=device, dtype=dtype)
    scaled = prior.node_attr.unsqueeze(-1) * noise
    x0 = torch.einsum("bij,bjd->bid", prior.evecs, scaled)
    x0 = torch.nan_to_num(x0)
    return center_coords(x0, node_mask)


def kabsch_align_batch_numpy(
    source: torch.Tensor,
    target: torch.Tensor,
    node_mask: torch.Tensor,
) -> torch.Tensor:
    """Align source to target per graph (NumPy fallback). Shapes [B, N, 3]."""
    aligned = source.clone()
    mask_f = node_mask.unsqueeze(-1).to(dtype=source.dtype)
    for g in range(source.shape[0]):
        m = node_mask[g].cpu().numpy()
        if not m.any():
            continue
        src_np = source[g, m].detach().cpu().numpy()
        tgt_np = target[g, m].detach().cpu().numpy()
        aligned_np = kabsch_align(src_np, tgt_np)
        aligned[g, m] = torch.as_tensor(aligned_np, device=source.device, dtype=source.dtype)
    return aligned * mask_f


def kabsch_align_batch_gpu(
    source: torch.Tensor,
    target: torch.Tensor,
    node_mask: torch.Tensor,
) -> torch.Tensor:
    """Batched Kabsch alignment on GPU. Shapes [B, N, 3]."""
    device_type = "cuda" if source.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        orig_dtype = source.dtype
        source = source.float()
        target = target.float()
        mask_f = node_mask.unsqueeze(-1).to(dtype=source.dtype)
        count = node_mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1)
        src_mean = (source * mask_f).sum(dim=1, keepdim=True) / count
        tgt_mean = (target * mask_f).sum(dim=1, keepdim=True) / count
        src_c = (source - src_mean) * mask_f
        tgt_c = (target - tgt_mean) * mask_f

        # Row-vector Kabsch alignment: src_c @ (U V^H).
        h = torch.einsum("bni,bnj->bij", src_c, tgt_c).float()
        u, _, vh = torch.linalg.svd(h)
        r = u @ vh

        det = torch.linalg.det(r)
        reflect = det < 0
        if reflect.any():
            u_fix = u.clone()
            u_fix[reflect, :, -1] *= -1
            r = u_fix @ vh

        aligned = (src_c @ r + tgt_mean).to(orig_dtype)
        return aligned * mask_f.to(dtype=orig_dtype)


def kabsch_align_batch(
    source: torch.Tensor,
    target: torch.Tensor,
    node_mask: torch.Tensor,
) -> torch.Tensor:
    """Align source to target per graph. Uses GPU when tensors are on CUDA."""
    if source.is_cuda:
        return kabsch_align_batch_gpu(source, target, node_mask)
    return kabsch_align_batch_numpy(source, target, node_mask)


def sample_flow_batch(
    x1: torch.Tensor,
    node_mask: torch.Tensor,
    sigma: float,
    *,
    prior_type: str = "gaussian",
    align_x0: bool = False,
    bond_features_dense: torch.Tensor | None = None,
    prior: PriorGraph | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample (x_t, t, x1) with dit_mc-compatible prior/alignment options."""
    device_type = "cuda" if x1.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        b = x1.shape[0]
        device = x1.device
        t = torch.empty(b, device=device).uniform_(0.0, 1.0)
        x1 = center_coords(x1.float(), node_mask)

        if prior_type == "harmonic":
            if bond_features_dense is None and prior is None:
                raise ValueError("bond_features_dense or prior required for harmonic prior")
            x0 = sample_harmonic_prior(bond_features_dense, node_mask, prior=prior)
        else:
            x0 = torch.randn_like(x1)
            x0 = center_coords(x0, node_mask)

        noise = torch.randn_like(x1)
        noise = center_coords(noise, node_mask)

        if align_x0:
            x0 = kabsch_align_batch(x0, x1, node_mask)
            x0 = center_coords(x0, node_mask)

        t_expand = t.view(b, 1, 1)
        x_t = (1 - t_expand) * x0 + t_expand * x1 + sigma * noise
        x_t = center_coords(x_t, node_mask)
    return x_t, t, x1


def sample_initial_coords(
    batch_size: int,
    num_atoms: int,
    node_mask: torch.Tensor,
    bond_features_dense: torch.Tensor,
    prior_type: str = "gaussian",
    device: torch.device | None = None,
    prior: PriorGraph | None = None,
) -> torch.Tensor:
    """Initial coordinates for ODE sampling."""
    device = device or bond_features_dense.device
    if prior_type == "harmonic":
        return sample_harmonic_prior(bond_features_dense, node_mask, prior=prior)
    x = torch.randn(batch_size, num_atoms, 3, device=device)
    return center_coords(x, node_mask)
