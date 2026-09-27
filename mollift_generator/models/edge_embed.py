"""DiTEdgeEmbed: 3D displacement RPE with optional RBF and shortest hops."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

HOP_NUM_EMBEDDINGS = 512


class DiTEdgeEmbed(nn.Module):
    """Encode pair features for attention rPE (dit_mc DiTEdgeEmbed style)."""

    def __init__(
        self,
        hidden_dim: int,
        num_radial_basis: int = 8,
        max_frequency: float = 2 * math.pi,
        embed_shortest_hops: bool = True,
        embed_distances: bool = True,
        radial_basis_bool: bool = False,
        hop_num_embeddings: int = HOP_NUM_EMBEDDINGS,
    ):
        super().__init__()
        self.num_radial_basis = num_radial_basis
        self.max_frequency = max_frequency
        self.embed_shortest_hops = embed_shortest_hops
        self.embed_distances = embed_distances
        self.radial_basis_bool = radial_basis_bool

        if radial_basis_bool:
            if num_radial_basis <= 1:
                raise ValueError("num_radial_basis must be > 1 when radial_basis_bool=True")
            disp_in = 3 * num_radial_basis
        else:
            disp_in = 3

        self.disp_mlp = nn.Sequential(
            nn.Linear(disp_in, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        if embed_shortest_hops:
            self.hop_embed = nn.Embedding(hop_num_embeddings, hidden_dim)
            self.hop_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            nn.init.zeros_(self.hop_mlp[-1].weight)
            nn.init.zeros_(self.hop_mlp[-1].bias)

    def _encode_displacement(self, disp: torch.Tensor) -> torch.Tensor:
        """Encode displacement vectors; disp ends with shape [..., 3]."""
        if self.radial_basis_bool:
            feat = self._radial_basis(disp)
        else:
            feat = disp
        return self.disp_mlp(feat)

    def _radial_basis(self, disp: torch.Tensor) -> torch.Tensor:
        """Fourier basis on displacement vectors [..., 3] -> [..., 3*num_radial_basis]."""
        limit = (self.num_radial_basis - 1) * math.pi / self.max_frequency
        freqs = torch.linspace(0, limit, self.num_radial_basis, device=disp.device, dtype=disp.dtype)
        prefix = disp.shape[:-1]
        scaled = disp.unsqueeze(-1) * freqs.view(*([1] * len(prefix)), 1, -1)
        return torch.sin(scaled).reshape(*prefix, 3 * self.num_radial_basis)

    def _encode_hops(self, shortest_path: torch.Tensor) -> torch.Tensor:
        hops = shortest_path.clamp(0, self.hop_embed.num_embeddings - 1)
        return self.hop_mlp(self.hop_embed(hops))

    def forward(
        self,
        x_t: torch.Tensor,
        shortest_path: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.embed_distances:
            # JAX: positions[src] - positions[dst]; dense[b,dst,src] = x[src] - x[dst]
            disp = x_t.unsqueeze(1) - x_t.unsqueeze(2)
            c_pair = self._encode_displacement(disp)
        else:
            c_pair = torch.zeros(
                *pair_mask.shape,
                self.disp_mlp[-1].out_features,
                device=x_t.device,
                dtype=x_t.dtype,
            )

        if self.embed_shortest_hops:
            c_pair = c_pair + self._encode_hops(shortest_path)
        return c_pair * pair_mask.unsqueeze(-1).to(c_pair.dtype)

    def forward_sparse(
        self,
        x_t: torch.Tensor,
        shortest_path: torch.Tensor,
        batch_idx: torch.Tensor,
        dst: torch.Tensor,
        src: torch.Tensor,
    ) -> torch.Tensor:
        """Returns edge features [E, H] on attention edges (dst=query, src=key)."""
        if self.embed_distances:
            disp = x_t[batch_idx, src] - x_t[batch_idx, dst]
            c_edge = self._encode_displacement(disp)
        else:
            c_edge = torch.zeros(
                batch_idx.shape[0],
                self.disp_mlp[-1].out_features,
                device=x_t.device,
                dtype=x_t.dtype,
            )

        if self.embed_shortest_hops:
            hops = shortest_path[batch_idx, dst, src]
            c_edge = c_edge + self._encode_hops(hops)
        return c_edge
