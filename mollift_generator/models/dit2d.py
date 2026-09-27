"""Mol2D3DFlow: top-level conditional flow matching model."""

from __future__ import annotations

import torch
import torch.nn as nn

from mollift_generator.models.edge_embed import DiTEdgeEmbed
from mollift_generator.models.graph_encoder import GraphEncoder
from mollift_generator.models.projection_encoder import ProjectionEncoder
from mollift_generator.models.time_embedding import TimeEmbedding
from mollift_generator.models.transformer_block import DiT2DBlock, modulate_adaln
from mollift_generator.utils.chirality import CHIRALITY_FEATURE_DIM
from mollift_generator.utils.sparse_graph import (
    bond_mask_to_edge_index,
    pair_mask_for_attention,
    pair_mask_to_edge_index,
)
from mollift_generator.utils.tensor_utils import center_coords


class Mol2D3DFlow(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        num_atom_types: int = 100,
        atom_feature_dim: int = 52,
        bond_feature_dim: int = 6,
        max_hop: int = 8,
        use_2d_condition: bool = True,
        use_absolute_pos_embed: bool = True,
        embed_distances: bool = False,
        embed_shortest_hops: bool = True,
        radial_basis_bool: bool = False,
        num_radial_basis: int = 8,
        use_pair_modulation: bool = True,
        act_dense_correct: bool = True,
        use_chirality: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_2d_condition = use_2d_condition
        self.use_chirality = use_chirality
        self.use_absolute_pos_embed = use_absolute_pos_embed
        self.embed_distances = embed_distances
        self.embed_shortest_hops = embed_shortest_hops
        self.use_pair_modulation = use_pair_modulation
        self.act_dense_correct = act_dense_correct

        self.atom_embed = nn.Embedding(num_atom_types, hidden_dim)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embed = TimeEmbedding(hidden_dim)
        self.graph_encoder = GraphEncoder(atom_feature_dim, bond_feature_dim, hidden_dim)
        self.projection_encoder = ProjectionEncoder(hidden_dim)
        if use_chirality:
            # Method B: per-atom target-chirality conditioning ([cw, ccw, is_stereocenter]).
            # Zero-init the output so a fresh chirality model starts identical to the base
            # (unconditional) model and learns to use the signal gradually.
            self.chirality_encoder = nn.Sequential(
                nn.Linear(CHIRALITY_FEATURE_DIM, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            nn.init.zeros_(self.chirality_encoder[-1].weight)
            nn.init.zeros_(self.chirality_encoder[-1].bias)
        self.edge_embed = DiTEdgeEmbed(
            hidden_dim,
            num_radial_basis=num_radial_basis,
            embed_shortest_hops=embed_shortest_hops,
            embed_distances=embed_distances,
            radial_basis_bool=radial_basis_bool,
        )
        self.cond_norm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList(
            [
                DiT2DBlock(
                    hidden_dim,
                    num_heads,
                    act_dense_correct=act_dense_correct,
                    use_pair_modulation=use_pair_modulation,
                )
                for _ in range(num_layers)
            ]
        )
        self.readout_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.readout_adaln = nn.Linear(hidden_dim, hidden_dim * 2)
        nn.init.zeros_(self.readout_adaln.weight)
        nn.init.zeros_(self.readout_adaln.bias)
        self.coord_head = nn.Linear(hidden_dim, 3)

    def _build_features_sparse(
        self,
        x_t: torch.Tensor,
        atom_features: torch.Tensor,
        bond_features_dense: torch.Tensor,
        shortest_path: torch.Tensor,
        pos_2d: torch.Tensor,
        node_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        proj_mask: torch.Tensor | None,
        attn_batch: torch.Tensor,
        attn_dst: torch.Tensor,
        attn_src: torch.Tensor,
        bond_batch: torch.Tensor,
        bond_dst: torch.Tensor,
        bond_src: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c_graph_atom = self.graph_encoder.forward_sparse(
            atom_features,
            bond_features_dense,
            node_mask,
            bond_batch,
            bond_dst,
            bond_src,
        )

        if self.use_2d_condition:
            c_2d_atom, c_2d_pair = self.projection_encoder.forward_sparse(
                pos_2d,
                node_mask,
                attn_batch,
                attn_dst,
                attn_src,
                proj_mask=proj_mask,
            )
        else:
            c_2d_atom = torch.zeros_like(c_graph_atom)
            c_2d_pair = torch.zeros(
                attn_batch.shape[0],
                self.hidden_dim,
                device=x_t.device,
                dtype=x_t.dtype,
            )

        features_edges = self.edge_embed.forward_sparse(
            x_t, shortest_path, attn_batch, attn_dst, attn_src
        ) + c_2d_pair

        return c_graph_atom, c_2d_atom, features_edges

    def _chirality_cond(
        self, chirality_feat: torch.Tensor | None, node_mask: torch.Tensor, ref: torch.Tensor
    ) -> torch.Tensor:
        """Per-atom chirality conditioning [B,N,H]; zeros when disabled/absent."""
        if not self.use_chirality or chirality_feat is None:
            return ref.new_zeros(ref.shape)
        c = self.chirality_encoder(chirality_feat.to(ref.dtype))
        return c * node_mask.unsqueeze(-1).to(c.dtype)

    def forward_dense(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        atom_types: torch.Tensor,
        atom_features: torch.Tensor,
        bond_features_dense: torch.Tensor,
        shortest_path: torch.Tensor,
        pos_2d: torch.Tensor,
        node_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        proj_mask: torch.Tensor | None = None,
        chirality_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dense reference forward for unit tests."""
        c_graph_atom = self.graph_encoder(atom_features, bond_features_dense, node_mask)

        if self.use_2d_condition:
            c_2d_atom, c_2d_pair = self.projection_encoder(pos_2d, node_mask, proj_mask)
        else:
            c_2d_atom = torch.zeros_like(c_graph_atom)
            c_2d_pair = torch.zeros(
                *shortest_path.shape,
                self.hidden_dim,
                device=x_t.device,
                dtype=x_t.dtype,
            )

        c_time = self.time_embed(t)
        if c_time.ndim == 1:
            c_time = c_time.unsqueeze(0)
        c_time = c_time.unsqueeze(1)

        c_chir_atom = self._chirality_cond(chirality_feat, node_mask, c_graph_atom)
        c_atom = self.cond_norm(c_time + c_graph_atom + c_2d_atom + c_chir_atom)
        attn_mask = pair_mask_for_attention(pair_mask)
        edge_feat = c_2d_pair + self.edge_embed(x_t, shortest_path, attn_mask)

        h = self.atom_embed(atom_types.clamp(min=0, max=self.atom_embed.num_embeddings - 1))
        if self.use_absolute_pos_embed:
            h = h + self.pos_mlp(x_t)
        h = h * node_mask.unsqueeze(-1).to(h.dtype)

        edge_for_attn = edge_feat if self.use_pair_modulation else None
        for block in self.blocks:
            h = block.forward_dense(h, c_atom, attn_mask, node_mask, edge_feat=edge_for_attn)

        shift, scale = self.readout_adaln(c_atom).chunk(2, dim=-1)
        h = modulate_adaln(self.readout_norm(h), shift, scale)
        x1_hat = self.coord_head(h)
        return center_coords(x1_hat, node_mask)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        atom_types: torch.Tensor,
        atom_features: torch.Tensor,
        bond_features_dense: torch.Tensor,
        shortest_path: torch.Tensor,
        pos_2d: torch.Tensor,
        node_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        proj_mask: torch.Tensor | None = None,
        chirality_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bond_batch, bond_dst, bond_src = bond_mask_to_edge_index(
            bond_features_dense, node_mask, exclude_self=True
        )
        attn_batch, attn_dst, attn_src = pair_mask_to_edge_index(pair_mask, exclude_self=True)

        c_graph_atom, c_2d_atom, features_edges = self._build_features_sparse(
            x_t,
            atom_features,
            bond_features_dense,
            shortest_path,
            pos_2d,
            node_mask,
            pair_mask,
            proj_mask,
            attn_batch,
            attn_dst,
            attn_src,
            bond_batch,
            bond_dst,
            bond_src,
        )

        c_time = self.time_embed(t)
        if c_time.ndim == 1:
            c_time = c_time.unsqueeze(0)
        c_time = c_time.unsqueeze(1)

        c_chir_atom = self._chirality_cond(chirality_feat, node_mask, c_graph_atom)
        c_atom = self.cond_norm(c_time + c_graph_atom + c_2d_atom + c_chir_atom)

        h = self.atom_embed(atom_types.clamp(min=0, max=self.atom_embed.num_embeddings - 1))
        if self.use_absolute_pos_embed:
            h = h + self.pos_mlp(x_t)
        h = h * node_mask.unsqueeze(-1).to(h.dtype)

        for block in self.blocks:
            h = block(
                h,
                c_atom,
                features_edges,
                attn_batch,
                attn_dst,
                attn_src,
                node_mask,
            )

        shift, scale = self.readout_adaln(c_atom).chunk(2, dim=-1)
        h = modulate_adaln(self.readout_norm(h), shift, scale)
        x1_hat = self.coord_head(h)
        return center_coords(x1_hat, node_mask)
