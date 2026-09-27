"""Graph encoders: dense MeshGraphNet-style and pair shortest-path encoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from mollift_generator.utils.sparse_graph import flat_segment_ids, segment_sum


class GraphEncoder(nn.Module):
    """Bond-graph message passing (sparse edges, dit_mc MGN style)."""

    def __init__(
        self,
        atom_feature_dim: int,
        bond_feature_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_in = nn.Linear(atom_feature_dim, hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_mlps = nn.ModuleList()
        self.node_mlps = nn.ModuleList()
        for _ in range(num_layers):
            self.edge_mlps.append(
                nn.Sequential(
                    nn.Linear(hidden_dim * 2 + bond_feature_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )
            self.node_mlps.append(
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )

    def forward_sparse(
        self,
        atom_features: torch.Tensor,
        bond_features_dense: torch.Tensor,
        node_mask: torch.Tensor,
        batch_idx: torch.Tensor,
        dst: torch.Tensor,
        src: torch.Tensor,
    ) -> torch.Tensor:
        """Returns c_graph_atom [B, N, H] via bond-edge message passing."""
        b, n, _ = atom_features.shape
        num_segments = b * n
        seg = flat_segment_ids(batch_idx, dst, n)

        h = self.node_norm(self.node_in(atom_features))
        bond_e = bond_features_dense[batch_idx, dst, src]
        for edge_mlp, node_mlp in zip(self.edge_mlps, self.node_mlps):
            h_dst = h[batch_idx, dst]
            h_src = h[batch_idx, src]
            m_e = edge_mlp(torch.cat([h_dst, h_src, bond_e], dim=-1))
            m_agg = segment_sum(m_e, seg, num_segments).view(b, n, self.hidden_dim)
            h = node_mlp(torch.cat([h, m_agg], dim=-1))
            h = h * node_mask.unsqueeze(-1).to(h.dtype)
        return h

    def forward(
        self,
        atom_features: torch.Tensor,
        bond_features_dense: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Dense fallback (tests / reference)."""
        h = self.node_norm(self.node_in(atom_features))
        pair_mask = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)

        for edge_mlp, node_mlp in zip(self.edge_mlps, self.node_mlps):
            hi = h.unsqueeze(2).expand(-1, -1, h.size(1), -1)
            hj = h.unsqueeze(1).expand(-1, h.size(1), -1, -1)
            edge_in = torch.cat([hi, hj, bond_features_dense], dim=-1)
            m_ij = edge_mlp(edge_in)
            m_ij = m_ij * pair_mask.unsqueeze(-1).to(m_ij.dtype)
            m_i = m_ij.sum(dim=2)
            h = node_mlp(torch.cat([h, m_i], dim=-1))
            h = h * node_mask.unsqueeze(-1).to(h.dtype)

        return h


class PairGraphEncoder(nn.Module):
    def __init__(self, hidden_dim: int, max_hop: int = 8):
        super().__init__()
        self.max_hop = max_hop
        self.embed = nn.Embedding(max_hop + 2, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, shortest_path: torch.Tensor) -> torch.Tensor:
        sp = shortest_path.clamp(0, self.max_hop + 1)
        return self.mlp(self.embed(sp))
