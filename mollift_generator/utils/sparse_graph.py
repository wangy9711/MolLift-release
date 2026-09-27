"""Sparse edge-list utilities for batched molecular graphs."""

from __future__ import annotations

import torch

from mollift_generator.utils.tensor_utils import pair_mask_from_node_mask


def pair_mask_for_attention(pair_mask: torch.Tensor) -> torch.Tensor:
    """Attention pair mask with self-loops removed (matches sparse edge list)."""
    n = pair_mask.shape[-1]
    eye = torch.eye(n, device=pair_mask.device, dtype=torch.bool)
    return pair_mask & ~eye.unsqueeze(0)


def pair_mask_to_edge_index(
    pair_mask: torch.Tensor,
    *,
    exclude_self: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert dense pair mask [B, N, N] to batched edge indices.

    Returns:
        batch_idx: [E] graph index in [0, B)
        dst: [E] center / receiver node (query) in [0, N)
        src: [E] neighbor / sender node (key) in [0, N)

    Edge semantics match dense SO3 attention: displacement is ``x[src] - x[dst]``,
    and attention aggregates incoming edges into ``dst`` nodes.
    """
    if pair_mask.dim() != 3:
        raise ValueError(f"pair_mask must be [B, N, N], got {tuple(pair_mask.shape)}")

    b, n, _ = pair_mask.shape
    mask = pair_mask
    if exclude_self:
        eye = torch.eye(n, device=pair_mask.device, dtype=torch.bool)
        mask = mask & ~eye.unsqueeze(0)

    batch_idx, dst, src = mask.nonzero(as_tuple=True)
    return batch_idx, dst, src


def flat_segment_ids(batch_idx: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Map (batch, dst) to a flat node id in [0, B * N)."""
    return batch_idx * num_nodes + dst


def segment_max(
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Per-segment maximum for 1D values [E]."""
    out = torch.full((num_segments,), float("-inf"), device=values.device, dtype=values.dtype)
    out.scatter_reduce_(0, segment_ids, values, reduce="amax", include_self=True)
    return out


def segment_sum(
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Sum values [E, ...] into segments; segment_ids is [E]."""
    feat_dim = values.shape[1:]
    out = torch.zeros((num_segments, *feat_dim), device=values.device, dtype=values.dtype)
    expand_ids = segment_ids.view(-1, *([1] * len(feat_dim))).expand_as(values)
    out.scatter_add_(0, expand_ids, values)
    return out


def segment_softmax(
    scores: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Softmax over variable-length edge groups sharing the same segment id.

    Args:
        scores: [E] or [E, H]
        segment_ids: [E]
        num_segments: total number of segments (typically B * N)
    """
    if scores.dim() == 1:
        max_scores = segment_max(scores, segment_ids, num_segments)
        stabilized = scores - max_scores[segment_ids]
        exp_scores = torch.exp(stabilized)
        denom = segment_sum(exp_scores, segment_ids, num_segments)
        return exp_scores / denom[segment_ids].clamp_min(1e-12)

    heads = []
    for h in range(scores.shape[1]):
        heads.append(segment_softmax(scores[:, h], segment_ids, num_segments))
    return torch.stack(heads, dim=1)


def gather_pair_features(
    tensor: torch.Tensor,
    batch_idx: torch.Tensor,
    dst: torch.Tensor,
    src: torch.Tensor,
) -> torch.Tensor:
    """Gather ``tensor[b, dst, src, ...]`` for each edge."""
    return tensor[batch_idx, dst, src]


def bond_mask_from_features(
    bond_features_dense: torch.Tensor,
    node_mask: torch.Tensor,
) -> torch.Tensor:
    """Bond adjacency mask [B, N, N] from bond type one-hot channels."""
    bond_mask = bond_features_dense[..., :4].sum(dim=-1) > 0
    return bond_mask & pair_mask_from_node_mask(node_mask)


def bond_mask_to_edge_index(
    bond_features_dense: torch.Tensor,
    node_mask: torch.Tensor,
    *,
    exclude_self: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Edge list over bonded atom pairs only."""
    return pair_mask_to_edge_index(
        bond_mask_from_features(bond_features_dense, node_mask),
        exclude_self=exclude_self,
    )
