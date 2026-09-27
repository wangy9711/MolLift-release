"""Distributed (DDP) training with validation and AMP."""

from __future__ import annotations

import argparse
import copy
import os
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from mollift_generator.data.collate import collate_molecules, collate_molecules_capped
from mollift_generator.data.geom_multiconformer_dataset import GEOMMultiConformerDataset
from mollift_generator.models.factory import build_model
from mollift_generator.training.config_utils import loss_config_from_cfg
from mollift_generator.training.ema import EMAModel
from mollift_generator.training.losses import compute_losses
from mollift_generator.training.lr_schedule import build_warmup_cosine_scheduler
from mollift_generator.training.train import set_seed
from mollift_generator.training.train_step import model_has_nan_params, optimizer_step
from mollift_generator.utils.config import load_config


def setup_distributed() -> tuple[int, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def build_dataset(
    cfg: dict, split: str, subset_indices=None
) -> GEOMMultiConformerDataset:
    dcfg = cfg["data"]
    dataset_name = dcfg.get("dataset", "qm9")
    if dataset_name in {"geom_qm9", "geom_drugs"}:
        if subset_indices is not None:
            raise ValueError("subset_path is only supported by the QM9-single dataset")
        limit_key = "train_molecule_limit" if split == dcfg["split"] else "valid_molecule_limit"
        return GEOMMultiConformerDataset(
            dcfg["cache_root"],
            split=split,
            conformer_sampling=dcfg.get("conformer_sampling", "uniform"),
            projection_mode=dcfg["projection_mode"],
            coord_units=dcfg.get("coord_units", "angstrom"),
            seed=cfg.get("seed", 42),
            molecule_limit=dcfg.get(limit_key),
        )
    raise ValueError(f"This trainer requires geom_drugs or geom_qm9, got {dataset_name}")


def build_ddp_dataloader(
    cfg: dict,
    split: str,
    world_size: int,
    rank: int,
    shuffle: bool,
    subset_indices=None,
) -> DataLoader:
    dcfg = cfg["data"]
    ds = build_dataset(cfg, split, subset_indices=subset_indices)
    sampler = (
        DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=shuffle)
        if world_size > 1
        else None
    )
    tcfg = cfg["train"]
    num_workers = tcfg.get("num_workers", 0)
    drop_last = shuffle and len(ds) >= tcfg["batch_size"] * max(world_size, 1)
    if dcfg.get("dynamic_padding", False):
        collate_fn = partial(collate_molecules_capped, max_atoms_cap=dcfg.get("max_atoms"))
    else:
        collate_fn = partial(collate_molecules, max_atoms=dcfg["max_atoms"])
    loader_kwargs = {
        "batch_size": tcfg["batch_size"],
        "collate_fn": collate_fn,
        "drop_last": drop_last,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["num_workers"] = num_workers
        # GEOM changes the selected conformer and projection every epoch. Disabling
        # persistent workers ensures worker-local Dataset copies receive set_epoch().
        loader_kwargs["persistent_workers"] = not isinstance(
            ds, GEOMMultiConformerDataset
        )
        loader_kwargs["prefetch_factor"] = tcfg.get("prefetch_factor", 4)
    if sampler is not None:
        loader_kwargs["sampler"] = sampler
    else:
        loader_kwargs["shuffle"] = shuffle
    return DataLoader(ds, **loader_kwargs)


def compute_epoch_train_loss(epoch_loss_sum: torch.Tensor, n_batches: int) -> float:
    """Average train loss; returns nan when every batch was skipped."""
    if n_batches == 0:
        return float("nan")
    return (epoch_loss_sum / n_batches).item()


@torch.no_grad()
def evaluate(
    model,
    loader,
    loss_cfg,
    device,
    use_amp: bool,
    proj_seed: int = 42,
) -> tuple[dict, int]:
    """Returns (metrics, num_skipped_batches). Skips non-finite batch losses."""
    model.eval()
    totals: dict[str, float] = {}
    n_batches = 0
    n_skipped = 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.autocast(device_type="cuda", enabled=use_amp and device.type == "cuda"):
            losses = compute_losses(
                model, batch, loss_cfg, deterministic_proj=True, proj_seed=proj_seed
            )
        if not torch.isfinite(losses["loss_total"]).all():
            n_skipped += 1
            continue
        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        n_batches += 1
    if n_batches == 0:
        nan_metrics = {k: float("nan") for k in ["loss_total", "loss_cfm", "loss_proj_xy"]}
        return nan_metrics, n_skipped
    return {k: v / n_batches for k, v in totals.items()}, n_skipped


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    cfg,
    epoch,
    val_loss,
    tag: str,
    ema: EMAModel | None = None,
    scheduler=None,
    scaler=None,
    global_step: int = 0,
    best_val: float = float("inf"),
):
    payload = {
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
        "epoch": epoch,
        "val_loss": val_loss,
        "tag": tag,
        "global_step": global_step,
        "best_val": best_val,
    }
    if ema is not None:
        payload["ema_state_dict"] = ema.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    torch.save(payload, path)


def train_ddp(cfg_path: str):
    cfg = load_config(cfg_path)
    rank, world_size, local_rank, device = setup_distributed()
    set_seed(cfg.get("seed", 42) + rank)

    tcfg = cfg["train"]
    dcfg = cfg["data"]
    use_amp = tcfg.get("use_amp", True)

    lock_path = dcfg.get("rdkit_lock")
    if lock_path:
        os.environ["MOLLIFT_RDKIT_LOCK"] = str(lock_path)
    elif dcfg.get("cache_dir"):
        os.environ.setdefault(
            "MOLLIFT_RDKIT_LOCK",
            str(Path(dcfg["cache_dir"]) / f".rdkit_{cfg.get('run_name', 'run')}.lock"),
        )

    subset_indices = None
    if dcfg.get("subset_path"):
        raise ValueError("Use train_molecule_limit/valid_molecule_limit for a GEOM smoke run")

    train_loader = build_ddp_dataloader(
        cfg, dcfg["split"], world_size, rank, shuffle=True, subset_indices=subset_indices
    )
    valid_split = dcfg.get("valid_split", "valid")
    if is_main(rank):
        valid_loader = build_ddp_dataloader(cfg, valid_split, 1, 0, shuffle=False)
    else:
        valid_loader = None

    train_ds = build_dataset(cfg, dcfg["split"], subset_indices=subset_indices)
    if is_main(rank):
        sample0 = collate_molecules([train_ds[0]], max_atoms=dcfg["max_atoms"])
        atom_dim = sample0["atom_features"].shape[-1]
        bond_dim = sample0["bond_features_dense"].shape[-1]
    else:
        atom_dim = 0
        bond_dim = 0

    dims = torch.tensor([atom_dim, bond_dim], dtype=torch.long, device=device)
    if world_size > 1:
        dist.broadcast(dims, src=0)
    atom_dim = int(dims[0].item())
    bond_dim = int(dims[1].item())
    cfg["model"]["atom_feature_dim"] = atom_dim
    cfg["model"]["bond_feature_dim"] = bond_dim

    model = build_model(cfg, atom_dim, bond_dim).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    peak_lr = tcfg["lr"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        weight_decay=tcfg.get("weight_decay", 0.0),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * tcfg["epochs"]
    scheduler = None
    if tcfg.get("lr_schedule", "constant") == "warmup_cosine":
        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            total_steps=total_steps,
            peak_lr=peak_lr,
            init_lr=tcfg.get("lr_init", 1e-5),
            warmup_fraction=tcfg.get("lr_warmup_fraction", 0.01),
            decay_fraction=tcfg.get("lr_decay_fraction", 0.99),
        )

    use_ema = tcfg.get("use_ema", False)
    ema = EMAModel(unwrap_model(model), decay=tcfg.get("ema_decay", 0.999)) if use_ema else None

    loss_cfg = loss_config_from_cfg(cfg, for_eval=False)
    eval_loss_cfg = loss_config_from_cfg(cfg, for_eval=True)

    ckpt_dir = Path(tcfg["checkpoint_dir"])
    log_dir = Path(tcfg.get("log_dir", "logs"))
    if is_main(rank):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

    run_name = cfg.get("run_name", "run")
    best_val = float("inf")
    global_step = 0
    start_epoch = 0
    latest_path = ckpt_dir / f"{run_name}_latest.pt"
    if latest_path.exists():
        if not tcfg.get("resume", False):
            raise FileExistsError(f"{latest_path} exists; enable train.resume or choose a new run_name")
        source = torch.load(latest_path, map_location="cpu", weights_only=False)
        # Resume restores state, not a new training schedule. Keep the configuration
        # (especially epochs/batch size/world size) unchanged for a continued run.
        unwrap_model(model).load_state_dict(source["model_state_dict"], strict=True)
        optimizer.load_state_dict(source["optimizer_state_dict"])
        if ema is not None:
            ema.load_state_dict(source["ema_state_dict"])
        if scheduler is not None:
            scheduler.load_state_dict(source["scheduler_state_dict"])
        if source.get("scaler_state_dict"):
            scaler.load_state_dict(source["scaler_state_dict"])
        start_epoch = int(source["epoch"])
        global_step = int(source.get("global_step", start_epoch * steps_per_epoch))
        best_val = float(source.get("best_val", source["val_loss"]))

    def log(msg: str):
        if is_main(rank):
            print(msg, flush=True)
            log_file = log_dir / f"{run_name}.log"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    grad_clip = tcfg.get("grad_clip_norm")
    nan_batches = 0
    skip_epoch_fraction = tcfg.get("skip_epoch_fraction", 0.5)

    for epoch in range(start_epoch, tcfg["epochs"]):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)

        model.train()
        epoch_loss_sum = torch.zeros((), device=device)
        n_batches = 0
        epoch_skips = 0
        total_batches = len(train_loader)
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp and device.type == "cuda"):
                losses = compute_losses(model, batch, loss_cfg)
            ok = optimizer_step(model, optimizer, scaler, losses["loss_total"], grad_clip)
            if not ok:
                epoch_skips += 1
                nan_batches += 1
                if is_main(rank):
                    log(
                        f"epoch={epoch+1} step={global_step+1} SKIP non-finite loss/grad "
                        f"cfm={losses['loss_cfm'].item():.4g} "
                        f"proj_xy={losses['loss_proj_xy'].item():.4g} "
                        f"bond={losses['loss_bond'].item():.4g} "
                        f"(epoch_skips={epoch_skips}/{total_batches}, total_skips={nan_batches})"
                    )
                if model_has_nan_params(model):
                    if is_main(rank):
                        log("Model parameters became non-finite; stopping training.")
                    cleanup_distributed()
                    return
                continue

            if ema is not None:
                ema.update(unwrap_model(model))
            if scheduler is not None:
                scheduler.step()

            epoch_loss_sum = epoch_loss_sum + losses["loss_total"].detach()
            n_batches += 1
            global_step += 1
            if is_main(rank) and global_step % tcfg.get("log_every", 100) == 0:
                lr = optimizer.param_groups[0]["lr"]
                log(
                    f"epoch={epoch+1} step={global_step} lr={lr:.2e} "
                    f"loss={losses['loss_total'].item():.4f} "
                    f"cfm={losses['loss_cfm'].item():.4f} "
                    f"proj_xy={losses['loss_proj_xy'].item():.4f} "
                    f"bond={losses['loss_bond'].item():.4f}"
                )

        avg_train = compute_epoch_train_loss(epoch_loss_sum, n_batches)
        val_skipped = 0
        if is_main(rank):
            if n_batches == 0:
                log(
                    f"Epoch {epoch+1}/{tcfg['epochs']}: all {total_batches} train batches skipped; "
                    f"train_loss=nan"
                )
            elif epoch_skips > 0:
                log(
                    f"Epoch {epoch+1}/{tcfg['epochs']}: skipped {epoch_skips}/{total_batches} "
                    f"train batches"
                )
            skip_frac = epoch_skips / max(total_batches, 1)
            if skip_frac > skip_epoch_fraction:
                log(
                    f"Early stop: {skip_frac:.0%} of epoch {epoch+1} batches skipped "
                    f"(threshold {skip_epoch_fraction:.0%})"
                )
                cleanup_distributed()
                return
            base = unwrap_model(model)
            saved_state = None
            if ema is not None:
                saved_state = copy.deepcopy(base.state_dict())
                ema.copy_to(base)
            validation_ratios = cfg["data"].get("validation_ratio_proj_choices")
            if validation_ratios:
                # Equal-weight validation across the configured observation ratios.
                metrics_by_ratio = []
                skipped_by_ratio = []
                for validation_ratio in validation_ratios:
                    ratio_loss_cfg = copy.copy(eval_loss_cfg)
                    ratio_loss_cfg.ratio_proj = float(validation_ratio)
                    ratio_metrics, ratio_skipped = evaluate(
                        base,
                        valid_loader,
                        ratio_loss_cfg,
                        device,
                        use_amp,
                        proj_seed=cfg.get("seed", 42),
                    )
                    metrics_by_ratio.append(ratio_metrics)
                    skipped_by_ratio.append(ratio_skipped)
                metric_names = set.intersection(*(set(metrics) for metrics in metrics_by_ratio))
                val_metrics = {
                    name: sum(metrics[name] for metrics in metrics_by_ratio) / len(metrics_by_ratio)
                    for name in metric_names
                }
                val_skipped = sum(skipped_by_ratio)
                val_ratio_losses = {
                    float(ratio): metrics["loss_total"]
                    for ratio, metrics in zip(validation_ratios, metrics_by_ratio)
                }
            else:
                val_metrics, val_skipped = evaluate(
                    base,
                    valid_loader,
                    eval_loss_cfg,
                    device,
                    use_amp,
                    proj_seed=cfg.get("seed", 42),
                )
                val_ratio_losses = None
            if saved_state is not None:
                base.load_state_dict(saved_state)
            val_loss = val_metrics["loss_total"]
        else:
            val_metrics = {}
            val_loss = 0.0

        if world_size > 1:
            val_tensor = torch.tensor(
                [val_loss if val_loss == val_loss else float("inf")],
                device=device,
            )
            dist.broadcast(val_tensor, src=0)
            val_loss = float(val_tensor.item())
            if val_loss == float("inf"):
                val_loss = float("nan")

        if is_main(rank):
            if not (avg_train == avg_train):
                log(
                    f"Epoch {epoch+1}/{tcfg['epochs']} non-finite train loss ({avg_train}); stopping."
                )
                cleanup_distributed()
                return

            val_finite = val_loss == val_loss
            skip_msg = f" val_skipped={val_skipped}" if val_skipped else ""
            train_finite = avg_train == avg_train
            if train_finite and val_finite:
                ratio_msg = (
                    " ratios="
                    + ",".join(f"{ratio:g}:{loss:.4f}" for ratio, loss in val_ratio_losses.items())
                    if val_ratio_losses is not None
                    else ""
                )
                log(
                    f"Epoch {epoch+1}/{tcfg['epochs']} train_loss={avg_train:.4f} "
                    f"val_loss={val_loss:.4f} val_cfm={val_metrics['loss_cfm']:.4f}"
                    f"{ratio_msg}{skip_msg}"
                )
            elif train_finite:
                log(
                    f"Epoch {epoch+1}/{tcfg['epochs']} train_loss={avg_train:.4f} "
                    f"val_loss=nan (skipped {val_skipped} batches); best_val={best_val:.4f}{skip_msg}"
                )
            elif val_finite:
                log(
                    f"Epoch {epoch+1}/{tcfg['epochs']} train_loss=nan "
                    f"val_loss={val_loss:.4f} val_cfm={val_metrics['loss_cfm']:.4f}{skip_msg}"
                )
            else:
                log(
                    f"Epoch {epoch+1}/{tcfg['epochs']} train_loss=nan "
                    f"val_loss=nan (skipped {val_skipped} batches); best_val={best_val:.4f}{skip_msg}"
                )

            improved = val_finite and val_loss < best_val
            if improved:
                best_val = val_loss
            save_checkpoint(
                ckpt_dir / f"{run_name}_latest.pt",
                model,
                optimizer,
                cfg,
                epoch + 1,
                val_loss,
                "latest",
                ema=ema,
                scheduler=scheduler,
                scaler=scaler, global_step=global_step, best_val=best_val,
            )
            if improved:
                best_val = val_loss
                save_checkpoint(
                    ckpt_dir / f"{run_name}_best.pt",
                    model,
                    optimizer,
                    cfg,
                    epoch + 1,
                    val_loss,
                    "best",
                    ema=ema,
                    scheduler=scheduler,
                    scaler=scaler, global_step=global_step, best_val=best_val,
                )
                log(f"New best val_loss={best_val:.4f} -> {run_name}_best.pt")

    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train_ddp(args.config)


if __name__ == "__main__":
    main()
