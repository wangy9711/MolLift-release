"""Euler ODE sampler for conditional flow matching."""

from __future__ import annotations

from typing import Dict

import torch

from mollift_generator.training.flow_utils import sample_initial_coords
from mollift_generator.utils.tensor_utils import center_coords


@torch.no_grad()
def euler_sample(
    model,
    batch: Dict[str, torch.Tensor],
    num_steps: int = 50,
    projection_guidance_alpha: float = 0.0,
    prior_type: str = "gaussian",
    eps: float = 1e-5,
    x_init: torch.Tensor | None = None,
) -> torch.Tensor:
    node_mask = batch["node_mask"]
    proj_mask = batch.get("proj_mask", node_mask)
    b, n, _ = batch["x3d"].shape
    device = batch["x3d"].device

    # Supplying x_init keeps prior draws independent of candidate batch size.
    if x_init is not None:
        x = x_init.to(device)
    else:
        x = sample_initial_coords(
            b,
            n,
            node_mask,
            batch["bond_features_dense"],
            prior_type=prior_type,
            device=device,
        )

    dt = 1.0 / num_steps
    for step in range(num_steps):
        t_val = step / num_steps
        t = torch.full((b,), t_val, device=device)
        x1_hat = model(
            x,
            t,
            batch["atom_types"],
            batch["atom_features"],
            batch["bond_features_dense"],
            batch["shortest_path"],
            batch["pos_2d"],
            node_mask,
            batch["pair_mask"],
            proj_mask,
            chirality_feat=batch.get("chirality_feat"),
        )
        v = (x1_hat - x) / (1.0 - t_val + eps)
        x = x + dt * v
        x = center_coords(x, node_mask)
        if projection_guidance_alpha > 0:
            alpha = projection_guidance_alpha
            obs = proj_mask.unsqueeze(-1)
            guided_xy = (1 - alpha) * x[..., :2] + alpha * batch["pos_2d"]
            x[..., :2] = torch.where(obs, guided_xy, x[..., :2])

    return x
