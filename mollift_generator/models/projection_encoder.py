"""2D projection conditioning encoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from mollift_generator.data.proj_mask import pair_mask_from_proj_mask


class ProjectionEncoder(nn.Module):
    def __init__(self, hidden_dim: int, invariant_only: bool = False):
        """Scalar 2D-projection conditioning.

        ``invariant_only=True`` emits only features that are invariant under the
        task's symmetry group (in-plane rotation about the viewing axis): the
        in-plane radius ``r`` per atom and the in-plane distance ``d2d`` per
        pair. The frame-dependent components (u, v, theta, du, dv) are dropped
        because, in the equivariant SO(3) model, absolute orientation must flow
        through the equivariant vector channel, not through invariant scalars.
        """
        super().__init__()
        self.invariant_only = invariant_only
        atom_in = 2 if invariant_only else 6
        pair_in = 1 if invariant_only else 3
        self.atom_mlp = nn.Sequential(
            nn.Linear(atom_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        pos_2d: torch.Tensor,
        node_mask: torch.Tensor,
        proj_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns c_2d_atom [B,N,H], c_2d_pair [B,N,N,H]."""
        if proj_mask is None:
            proj_mask = node_mask
        obs = proj_mask & node_mask
        obs_f = obs.to(dtype=pos_2d.dtype)

        u = pos_2d[..., 0] * obs_f
        v = pos_2d[..., 1] * obs_f
        r = torch.sqrt(u**2 + v**2 + 1e-8)
        if self.invariant_only:
            atom_feat = torch.stack([r, node_mask.to(u.dtype)], dim=-1)
        else:
            theta = torch.atan2(v, u)
            atom_feat = torch.stack(
                [u, v, r, torch.sin(theta), torch.cos(theta), node_mask.to(u.dtype)],
                dim=-1,
            )
        c_atom = self.atom_mlp(atom_feat) * obs.unsqueeze(-1).to(atom_feat.dtype)

        du = u.unsqueeze(-1) - u.unsqueeze(-2)
        dv = v.unsqueeze(-1) - v.unsqueeze(-2)
        d2d = torch.sqrt(du**2 + dv**2 + 1e-8)
        if self.invariant_only:
            pair_feat = d2d.unsqueeze(-1)
        else:
            pair_feat = torch.stack([du, dv, d2d], dim=-1)
        c_pair = self.pair_mlp(pair_feat)
        proj_pair_mask = pair_mask_from_proj_mask(obs)
        c_pair = c_pair * proj_pair_mask.unsqueeze(-1).to(c_pair.dtype)
        return c_atom, c_pair

    def forward_sparse(
        self,
        pos_2d: torch.Tensor,
        node_mask: torch.Tensor,
        batch_idx: torch.Tensor,
        dst: torch.Tensor,
        src: torch.Tensor,
        proj_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns c_2d_atom [B,N,H] and c_2d_pair [E,H] on attention edges."""
        if proj_mask is None:
            proj_mask = node_mask
        obs = proj_mask & node_mask
        obs_f = obs.to(dtype=pos_2d.dtype)

        u = pos_2d[..., 0] * obs_f
        v = pos_2d[..., 1] * obs_f
        r = torch.sqrt(u**2 + v**2 + 1e-8)
        if self.invariant_only:
            atom_feat = torch.stack([r, node_mask.to(u.dtype)], dim=-1)
        else:
            theta = torch.atan2(v, u)
            atom_feat = torch.stack(
                [u, v, r, torch.sin(theta), torch.cos(theta), node_mask.to(u.dtype)],
                dim=-1,
            )
        c_atom = self.atom_mlp(atom_feat) * obs.unsqueeze(-1).to(atom_feat.dtype)

        du = u[batch_idx, dst] - u[batch_idx, src]
        dv = v[batch_idx, dst] - v[batch_idx, src]
        d2d = torch.sqrt(du**2 + dv**2 + 1e-8)
        if self.invariant_only:
            pair_feat = d2d.unsqueeze(-1)
        else:
            pair_feat = torch.stack([du, dv, d2d], dim=-1)
        c_pair = self.pair_mlp(pair_feat)

        obs_dst = obs[batch_idx, dst]
        obs_src = obs[batch_idx, src]
        c_pair = c_pair * (obs_dst & obs_src).unsqueeze(-1).to(c_pair.dtype)
        return c_atom, c_pair
