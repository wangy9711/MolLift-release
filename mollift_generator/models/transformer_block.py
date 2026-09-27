"""DiT block with AdaLN, AdaScale, and pair-conditioned attention."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from mollift_generator.utils.sparse_graph import flat_segment_ids, pair_mask_for_attention, segment_softmax, segment_sum


def modulate_adaln(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class AdaLNModulation(nn.Module):
    """Produce shift, scale, gate from per-atom conditioning."""

    def __init__(self, hidden_dim: int, num_groups: int = 3, act_dense_correct: bool = True):
        super().__init__()
        self.num_groups = num_groups
        self.act_dense_correct = act_dense_correct
        self.linear = nn.Linear(hidden_dim, hidden_dim * 3 * num_groups)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, c_atom: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.act_dense_correct:
            params = self.linear(torch.nn.functional.silu(c_atom))
        else:
            params = self.linear(c_atom)
        chunks = params.chunk(self.num_groups * 3, dim=-1)
        return tuple(chunks)


class PairConditionedAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        use_pair_modulation: bool = True,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_pair_modulation = use_pair_modulation
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.bias_mlp = nn.Sequential(
            nn.Linear(hidden_dim, num_heads),
            nn.SiLU(),
            nn.Linear(num_heads, num_heads),
        )
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def _attention_pair_mask(self, pair_mask: torch.Tensor) -> torch.Tensor:
        return pair_mask_for_attention(pair_mask)

    def _forward_standard(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_mask = self._attention_pair_mask(pair_mask)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~attn_mask.unsqueeze(1), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        return torch.matmul(attn, v)

    def _forward_modulation_dense(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        edge_feat: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Dense rPE modulation (reference / tests). edge_feat: [B,N,N,H]."""
        b, _, n, _ = q.shape
        attn_mask = self._attention_pair_mask(pair_mask)
        u = edge_feat.reshape(b, n, n, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        q_i = q.unsqueeze(3)
        k_j = k.unsqueeze(2)
        v_j = v.unsqueeze(2)
        k_mod = k_j * u
        scores = (q_i * k_mod).sum(-1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~attn_mask.unsqueeze(1), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        v_mod = v_j * u
        return (attn.unsqueeze(-1) * v_mod).sum(dim=3)

    def forward_modulation_dense(
        self,
        h: torch.Tensor,
        edge_feat: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Dense modulation forward for equivalence tests."""
        b, n, d = h.shape
        qkv = self.qkv(h).reshape(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = self._forward_modulation_dense(q, k, v, edge_feat, pair_mask)
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.out_proj(out)

    def forward_sparse(
        self,
        h: torch.Tensor,
        edge_feat: torch.Tensor,
        batch_idx: torch.Tensor,
        dst: torch.Tensor,
        src: torch.Tensor,
    ) -> torch.Tensor:
        """Sparse rPE modulation over edge list (production path)."""
        b, n, d = h.shape
        num_segments = b * n
        seg = flat_segment_ids(batch_idx, dst, n)

        qkv = self.qkv(h).reshape(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        q_dst = q[batch_idx, dst]
        k_src = k[batch_idx, src]
        v_src = v[batch_idx, src]

        u = edge_feat.reshape(-1, self.num_heads, self.head_dim)
        k_mod = k_src * u
        v_mod = v_src * u
        scores = (q_dst * k_mod).sum(-1) / math.sqrt(self.head_dim)
        attn = segment_softmax(scores, seg, num_segments)
        weighted_v = attn.unsqueeze(-1) * v_mod
        out_flat = segment_sum(weighted_v, seg, num_segments)
        out = out_flat.view(b, n, d)
        return self.out_proj(out)

    def forward(
        self,
        h: torch.Tensor,
        pair_mask: torch.Tensor,
        edge_feat: torch.Tensor | None = None,
        batch_idx: torch.Tensor | None = None,
        dst: torch.Tensor | None = None,
        src: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if batch_idx is not None:
            if edge_feat is None or dst is None or src is None:
                raise ValueError("edge_feat, dst, and src are required for sparse attention")
            return self.forward_sparse(h, edge_feat, batch_idx, dst, src)

        b, n, d = h.shape
        qkv = self.qkv(h).reshape(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_pair_modulation:
            if edge_feat is None:
                raise ValueError("edge_feat is required for pair modulation attention")
            out = self._forward_modulation_dense(q, k, v, edge_feat, pair_mask)
        else:
            out = self._forward_standard(q, k, v, pair_mask)

        out = out.transpose(1, 2).reshape(b, n, d)
        return self.out_proj(out)


class DiT2DBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        act_dense_correct: bool = True,
        use_pair_modulation: bool = True,
    ):
        super().__init__()
        self.use_pair_modulation = use_pair_modulation
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.adaln1 = AdaLNModulation(hidden_dim, num_groups=1, act_dense_correct=act_dense_correct)
        self.adaln2 = AdaLNModulation(hidden_dim, num_groups=1, act_dense_correct=act_dense_correct)
        self.attn = PairConditionedAttention(hidden_dim, num_heads, use_pair_modulation=use_pair_modulation)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_dim),
        )

    def forward_sparse(
        self,
        h: torch.Tensor,
        c_atom: torch.Tensor,
        edge_feat: torch.Tensor,
        batch_idx: torch.Tensor,
        dst: torch.Tensor,
        src: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        shift1, scale1, gate1 = self.adaln1(c_atom)
        h_norm = modulate_adaln(self.norm1(h), shift1, scale1)
        h = h + gate1 * self.attn.forward_sparse(h_norm, edge_feat, batch_idx, dst, src)

        shift2, scale2, gate2 = self.adaln2(c_atom)
        h_norm = modulate_adaln(self.norm2(h), shift2, scale2)
        h = h + gate2 * self.mlp(h_norm)
        return h * node_mask.unsqueeze(-1).to(h.dtype)

    def forward_dense(
        self,
        h: torch.Tensor,
        c_atom: torch.Tensor,
        pair_mask: torch.Tensor,
        node_mask: torch.Tensor,
        edge_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dense reference path for unit tests."""
        shift1, scale1, gate1 = self.adaln1(c_atom)
        h_norm = modulate_adaln(self.norm1(h), shift1, scale1)
        h = h + gate1 * self.attn(h_norm, pair_mask, edge_feat=edge_feat)

        shift2, scale2, gate2 = self.adaln2(c_atom)
        h_norm = modulate_adaln(self.norm2(h), shift2, scale2)
        h = h + gate2 * self.mlp(h_norm)
        return h * node_mask.unsqueeze(-1).to(h.dtype)

    def forward(
        self,
        h: torch.Tensor,
        c_atom: torch.Tensor,
        edge_feat: torch.Tensor,
        batch_idx: torch.Tensor,
        dst: torch.Tensor,
        src: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_sparse(h, c_atom, edge_feat, batch_idx, dst, src, node_mask)
