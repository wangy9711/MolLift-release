"""Training losses: CFM and projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch

from mollift_generator.data.proj_mask import (
    pair_mask_from_proj_mask,
    sample_proj_mask_deterministic,
    sample_proj_mask_random,
    sample_proj_mask_random_choices,
)
from mollift_generator.training.augmentation import apply_rotation_augmentation
from mollift_generator.training.flow_utils import PriorGraph, sample_flow_batch
from mollift_generator.utils.sparse_graph import bond_mask_from_features
from mollift_generator.utils.tensor_utils import masked_mean


@dataclass
class LossConfig:
    sigma: float = 0.05
    t_weight_cap: float = 0.9
    lambda_proj_xy: float = 1.0
    lambda_proj_pair: float = 0.0
    lambda_bond: float = 1.0
    ratio_proj: float = 1.0
    ratio_proj_choices: tuple[float, ...] | None = None
    prior_type: str = "gaussian"
    align_x0: bool = False
    augmentation_bool: bool = False
    chirality_dropout: float = 0.0


def apply_chirality_dropout(
    chirality_feat: torch.Tensor, p: float, *, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Per-molecule CFG-style dropout of the target-chirality handedness.

    With prob ``p`` per molecule, zero the [cw, ccw] channels (keep is_stereocenter), i.e.
    turn its provided stereocenters into the UNKNOWN token so the model also learns the
    unconditional (handedness-agnostic) behavior. p=0 -> unchanged; p=1 -> all dropped.
    """
    if p <= 0.0:
        return chirality_feat
    b = chirality_feat.shape[0]
    drop = torch.rand(b, generator=generator, device=chirality_feat.device) < p
    if not drop.any():
        return chirality_feat
    out = chirality_feat.clone()
    m = drop.view(b, 1, 1)
    out[..., 0:2] = torch.where(m, torch.zeros_like(out[..., 0:2]), out[..., 0:2])
    return out


def chirality_to_unknown(chirality_feat: torch.Tensor) -> torch.Tensor:
    """Deterministically downgrade every atom's handedness to the UNKNOWN token.

    Zeros the [cw, ccw] channels and keeps is_stereocenter -- the p=1 limit of
    ``apply_chirality_dropout``. This puts a Method-B model into its unconditional
    (handedness-agnostic) regime, so ONE checkpoint can be scored both ways for the
    2.2 A-vs-B decision: chirality fed (=B) vs not fed (=needs post-hoc Method A).
    """
    out = chirality_feat.clone()
    out[..., 0:2] = 0.0
    return out


def build_proj_mask(
    node_mask: torch.Tensor,
    ratio_proj: float,
    *,
    ratio_proj_choices: Sequence[float] | None = None,
    deterministic: bool = False,
    sample_keys: Optional[torch.Tensor] = None,
    proj_seed: int = 42,
) -> torch.Tensor:
    batch_size = node_mask.shape[0]
    if sample_keys is None:
        keys = list(range(batch_size))
    elif (
        deterministic
        and sample_keys.device.type == "cpu"
        and sample_keys.numel() == batch_size
        and torch.equal(sample_keys, torch.arange(batch_size, dtype=sample_keys.dtype))
    ):
        keys = list(range(batch_size))
    elif sample_keys.device.type == "cpu":
        keys = [int(k) for k in sample_keys.tolist()]
    else:
        keys = sample_keys.detach().cpu().tolist()
    if deterministic:
        return sample_proj_mask_deterministic(node_mask, ratio_proj, keys, seed=proj_seed)
    if ratio_proj_choices is not None:
        return sample_proj_mask_random_choices(node_mask, ratio_proj_choices)
    return sample_proj_mask_random(node_mask, ratio_proj)


def loss_cfm(
    x1_hat: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
    node_mask: torch.Tensor,
    t_weight_cap: float = 0.9,
) -> torch.Tensor:
    weight = 1.0 / (1.0 - torch.clamp(t, max=t_weight_cap) + 1e-4).pow(2)
    sq = ((x1_hat - x1) ** 2).mean(dim=-1)
    per_graph = masked_mean(sq * weight.view(-1, 1), node_mask, dim=1)
    return per_graph.mean()


def loss_proj_xy(
    x1_hat: torch.Tensor,
    pos_2d: torch.Tensor,
    proj_mask: torch.Tensor,
) -> torch.Tensor:
    if not proj_mask.any():
        return x1_hat.new_zeros(())
    diff = (x1_hat[..., :2] - pos_2d) ** 2
    return masked_mean(diff.sum(dim=-1), proj_mask)


def loss_proj_pair(
    x1_hat: torch.Tensor,
    pos_2d: torch.Tensor,
    proj_mask: torch.Tensor,
) -> torch.Tensor:
    pair_mask = pair_mask_from_proj_mask(proj_mask)
    if not pair_mask.any():
        return x1_hat.new_zeros(())
    with torch.autocast(device_type=x1_hat.device.type, enabled=False):
        xy_hat = x1_hat[..., :2].float()
        pos_2d_f = pos_2d.float()
        d_hat = torch.norm(xy_hat.unsqueeze(2) - xy_hat.unsqueeze(1), dim=-1)
        d_ref = torch.norm(pos_2d_f.unsqueeze(2) - pos_2d_f.unsqueeze(1), dim=-1)
        sq = (d_hat - d_ref) ** 2
    return masked_mean(sq, pair_mask)


def loss_bond(
    x1_hat: torch.Tensor,
    x1: torch.Tensor,
    bond_features_dense: torch.Tensor,
    node_mask: torch.Tensor,
) -> torch.Tensor:
    bond_mask = bond_mask_from_features(bond_features_dense, node_mask)
    if not bond_mask.any():
        return x1_hat.new_zeros(())
    with torch.autocast(device_type=x1_hat.device.type, enabled=False):
        x_hat = x1_hat.float()
        x_true = x1.float()
        d_hat = torch.norm(x_hat.unsqueeze(2) - x_hat.unsqueeze(1), dim=-1)
        d_true = torch.norm(x_true.unsqueeze(2) - x_true.unsqueeze(1), dim=-1)
        sq = (d_hat - d_true) ** 2
    return masked_mean(sq, bond_mask)


def compute_losses(
    model,
    batch: Dict[str, torch.Tensor],
    cfg: LossConfig,
    *,
    deterministic_proj: bool = False,
    proj_seed: int = 42,
) -> Dict[str, torch.Tensor]:
    if cfg.augmentation_bool:
        batch = apply_rotation_augmentation(batch)

    node_mask = batch["node_mask"]
    proj_mask = batch.get("proj_mask")
    if proj_mask is None:
        proj_mask = build_proj_mask(
            node_mask,
            cfg.ratio_proj,
            ratio_proj_choices=cfg.ratio_proj_choices,
            deterministic=deterministic_proj,
            sample_keys=batch.get("sample_keys"),
            proj_seed=proj_seed,
        )

    prior = None
    if "prior_node_attr" in batch and "prior_evecs" in batch:
        prior = PriorGraph(
            node_attr=batch["prior_node_attr"],
            evecs=batch["prior_evecs"],
        )

    chirality_feat = batch.get("chirality_feat")
    if chirality_feat is not None and not deterministic_proj:
        # Only drop during (stochastic) training, never during deterministic eval.
        chirality_feat = apply_chirality_dropout(chirality_feat, cfg.chirality_dropout)

    x_t, t, x1 = sample_flow_batch(
        batch["x3d"],
        node_mask,
        cfg.sigma,
        prior_type=cfg.prior_type,
        align_x0=cfg.align_x0,
        bond_features_dense=batch["bond_features_dense"],
        prior=prior,
    )
    x1_hat = model(
        x_t,
        t,
        batch["atom_types"],
        batch["atom_features"],
        batch["bond_features_dense"],
        batch["shortest_path"],
        batch["pos_2d"],
        node_mask,
        batch["pair_mask"],
        proj_mask,
        chirality_feat=chirality_feat,
    )

    losses = {
        "loss_cfm": loss_cfm(x1_hat, x1, t, node_mask, cfg.t_weight_cap),
        "loss_proj_xy": loss_proj_xy(x1_hat, batch["pos_2d"], proj_mask),
        "loss_proj_pair": loss_proj_pair(x1_hat, batch["pos_2d"], proj_mask),
        "loss_bond": loss_bond(x1_hat, x1, batch["bond_features_dense"], node_mask),
    }
    losses["loss_total"] = (
        losses["loss_cfm"]
        + cfg.lambda_proj_xy * losses["loss_proj_xy"]
        + cfg.lambda_proj_pair * losses["loss_proj_pair"]
        + cfg.lambda_bond * losses["loss_bond"]
    )
    return losses
