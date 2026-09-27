"""Build LossConfig from train/data config sections."""

from __future__ import annotations

from mollift_generator.training.losses import LossConfig


def loss_config_from_cfg(cfg: dict, *, for_eval: bool = False) -> LossConfig:
    tcfg = cfg["train"]
    dcfg = cfg["data"]
    return LossConfig(
        sigma=tcfg["sigma"],
        t_weight_cap=tcfg.get("t_weight_cap", 0.9),
        lambda_proj_xy=tcfg["lambda_proj_xy"],
        lambda_proj_pair=tcfg.get("lambda_proj_pair", 0.0),
        lambda_bond=tcfg.get("lambda_bond", 1.0),
        ratio_proj=dcfg.get("ratio_proj", 1.0),
        # Each validation pass uses a fixed ratio; random ratios apply only to training.
        ratio_proj_choices=(
            None
            if for_eval or dcfg.get("ratio_proj_choices") is None
            else tuple(float(value) for value in dcfg["ratio_proj_choices"])
        ),
        prior_type=tcfg.get("prior_type", "gaussian"),
        align_x0=tcfg.get("align_x0", False),
        augmentation_bool=tcfg.get("augmentation_bool", False) and not for_eval,
        chirality_dropout=(0.0 if for_eval else tcfg.get("chirality_dropout", 0.0)),
    )
