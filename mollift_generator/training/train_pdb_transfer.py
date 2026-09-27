"""Single-GPU GEOM-Drugs -> PDB-fragment transfer training."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mollift_generator.data.collate import collate_molecules_capped
from mollift_generator.data.geom_cache import read_records
from mollift_generator.data.geom_multiconformer_dataset import GEOMMultiConformerDataset
from mollift_generator.data.pdb_fragment_dataset import (
    PDBDrugsReplayDataset,
    PDBFragmentDataset,
    TransferBucketBatchSampler,
)
from mollift_generator.models.factory import build_model
from mollift_generator.training.config_utils import loss_config_from_cfg
from mollift_generator.training.ema import EMAModel
from mollift_generator.training.losses import compute_losses
from mollift_generator.training.lr_schedule import build_warmup_cosine_scheduler
from mollift_generator.training.train import set_seed
from mollift_generator.training.train_ddp import evaluate
from mollift_generator.utils.config import load_config


def _bucket_batches(
    n_atoms: np.ndarray, buckets: list[tuple[int, int]], group_keys: np.ndarray | None = None
) -> list[list[int]]:
    result = []
    lower = 0
    for upper, batch_size in buckets:
        indices = np.flatnonzero((n_atoms > lower) & (n_atoms <= upper)).tolist()
        if group_keys is not None:
            indices.sort(key=lambda index: (group_keys[index], index))
        result.extend(indices[offset:offset + batch_size] for offset in range(0, len(indices), batch_size))
        lower = upper
    return result


def _loader(dataset, *, batches=None, batch_size=None, workers=0):
    kwargs = {
        "dataset": dataset, "collate_fn": collate_molecules_capped,
        "pin_memory": torch.cuda.is_available(), "num_workers": workers,
    }
    if batches is not None:
        kwargs["batch_sampler"] = batches
    else:
        kwargs.update(batch_size=batch_size, shuffle=False)
    if workers > 0:
        kwargs["persistent_workers"] = False
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def _wait_for_training_gpu(max_used_mib: int, poll_seconds: int = 60) -> None:
    """Gate immediately before model allocation, after CPU data preparation.

    ``CUDA_VISIBLE_DEVICES`` has already selected the physical card, so CUDA device 0 is
    the intended training card.  Querying through torch avoids fragile physical-index
    parsing and leaves only a very small interval between gate release and model allocation.
    """
    if max_used_mib <= 0:
        return
    while True:
        free_bytes, total_bytes = torch.cuda.mem_get_info(torch.device("cuda:0"))
        used_mib = (total_bytes - free_bytes) / (1024 ** 2)
        if used_mib <= max_used_mib:
            print(
                f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                f"training GPU ready after data preparation: {used_mib:.0f} MiB used; "
                "allocating model now",
                flush=True,
            )
            return
        print(
            f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
            f"data ready; waiting immediately before model allocation: "
            f"{used_mib:.0f} MiB used > {max_used_mib} MiB",
            flush=True,
        )
        time.sleep(max(1, poll_seconds))


@torch.no_grad()
def _evaluate_ratios(model, loader, cfg, device, use_amp) -> tuple[float, dict[float, float]]:
    ratios = cfg["data"].get("validation_ratio_proj_choices") or [cfg["data"].get("ratio_proj", 1.0)]
    base = loss_config_from_cfg(cfg, for_eval=True)
    values = {}
    for ratio in ratios:
        ratio_cfg = copy.copy(base); ratio_cfg.ratio_proj = float(ratio)
        metrics, skipped = evaluate(model, loader, ratio_cfg, device, use_amp, proj_seed=cfg.get("seed", 42))
        if skipped:
            raise RuntimeError(f"validation skipped {skipped} non-finite batches at ratio={ratio}")
        values[float(ratio)] = float(metrics["loss_total"])
    return sum(values.values()) / len(values), values


def _load_transfer_weights(model, path: str | Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("ema_state_dict")
    if not state:
        raise ValueError(f"transfer checkpoint has no EMA state: {path}")
    if not any(key.endswith("time_embed.freqs") for key in state):
        raise ValueError(f"transfer checkpoint is missing time_embed.freqs: {path}")
    model.load_state_dict(state, strict=True)
    return checkpoint


def _save_transfer_checkpoint(
    path: Path, model, optimizer, scheduler, ema, cfg, *, epoch: int,
    val_loss: float, tag: str, global_step: int, transfer_baseline: dict,
) -> None:
    """Save all state needed to resume without moving the retention baseline."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "config": cfg,
        "epoch": epoch,
        "val_loss": val_loss,
        "tag": tag,
        "global_step": global_step,
        "transfer_baseline": transfer_baseline,
    }, path)


def train(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    if not torch.cuda.is_available():
        raise RuntimeError("PDB transfer training requires CUDA")
    device = torch.device("cuda:0")
    dcfg, tcfg = cfg["data"], cfg["train"]
    buckets = [(int(item["max_atoms"]), int(item["batch_size"])) for item in tcfg["atom_buckets"]]

    pdb_train = PDBFragmentDataset(
        dcfg["pdb_root"], "train", seed=cfg.get("seed", 42),
        limit_per_stratum=dcfg.get("pdb_train_per_stratum"),
    )
    pdb_valid = PDBFragmentDataset(
        dcfg["pdb_root"], "valid", seed=cfg.get("seed", 42),
        limit_per_stratum=dcfg.get("pdb_valid_per_stratum", 10),
    )
    drugs_train = GEOMMultiConformerDataset(
        dcfg["drugs_cache_root"], split="train", conformer_sampling="uniform",
        projection_mode="synthetic", coord_units="angstrom", seed=cfg.get("seed", 42),
        molecule_limit=dcfg.get("drugs_train_limit"),
    )
    drugs_valid = GEOMMultiConformerDataset(
        dcfg["drugs_cache_root"], split="valid", conformer_sampling="uniform",
        projection_mode="synthetic", coord_units="angstrom", seed=cfg.get("seed", 42),
        molecule_limit=dcfg.get("drugs_valid_limit", 2000),
    )
    union = PDBDrugsReplayDataset(pdb_train, drugs_train)
    drug_atoms = [record.n_atoms for record in drugs_train.records]
    curriculum = [(int(item["until_epoch"]), int(item["max_length"])) for item in tcfg.get("curriculum", [])]
    batch_sampler = TransferBucketBatchSampler(
        pdb_train, drug_atoms, pdb_samples_per_epoch=tcfg["pdb_samples_per_epoch"],
        pdb_fraction=dcfg.get("pdb_fraction", 0.75), bucket_batch_sizes=buckets,
        seed=cfg.get("seed", 42), curriculum=curriculum,
    )
    workers = int(tcfg.get("num_workers", 0))
    train_loader = _loader(union, batches=batch_sampler, workers=workers)
    valid_batches = _bucket_batches(
        pdb_valid.n_atoms, buckets, pdb_valid.manifest.shard.to_numpy()
    )
    pdb_valid_loader = _loader(pdb_valid, batches=valid_batches, workers=workers)
    drugs_valid_loader = _loader(drugs_valid, batch_size=tcfg.get("drugs_valid_batch_size", 16), workers=workers)

    _wait_for_training_gpu(
        int(tcfg.get("gpu_max_used_mib", 0)),
        int(tcfg.get("gpu_gate_poll_seconds", 60)),
    )

    checkpoint_dir = Path(tcfg["checkpoint_dir"]); checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(tcfg["log_dir"]); log_dir.mkdir(parents=True, exist_ok=True)
    run_name = cfg["run_name"]
    latest_path = checkpoint_dir / f"{run_name}_latest.pt"
    resume = bool(tcfg.get("resume", True)) and latest_path.exists()

    model = build_model(cfg, atom_feature_dim=72, bond_feature_dim=5).to(device)
    if resume:
        source = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(source["model_state_dict"], strict=True)
    else:
        source = _load_transfer_weights(model, tcfg["init_checkpoint"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.0))
    target_graphs = int(tcfg.get("target_graphs_per_step", 32))
    optimizer_steps_per_epoch = math.ceil(
        (tcfg["pdb_samples_per_epoch"] / dcfg.get("pdb_fraction", 0.75)) / target_graphs
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer, total_steps=max(1, optimizer_steps_per_epoch * tcfg["epochs"]),
        peak_lr=tcfg["lr"], init_lr=tcfg.get("lr_init", 1e-6),
        warmup_fraction=tcfg.get("lr_warmup_fraction", 0.03),
        decay_fraction=tcfg.get("lr_decay_fraction", 0.97),
    )
    ema = EMAModel(model, decay=tcfg.get("ema_decay", 0.999))
    start_epoch = 0
    global_step = 0
    if resume:
        optimizer.load_state_dict(source["optimizer_state_dict"])
        scheduler.load_state_dict(source["scheduler_state_dict"])
        ema.load_state_dict(source["ema_state_dict"])
        start_epoch = int(source["epoch"])
        global_step = int(source.get("global_step", scheduler.last_epoch))
    use_amp = bool(tcfg.get("use_amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_cfg = loss_config_from_cfg(cfg, for_eval=False)
    grad_clip = float(tcfg.get("grad_clip_norm", 1.0))

    log_path = log_dir / f"{run_name}.log"
    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a") as handle:
            handle.write(message + "\n")

    log(json.dumps({
        "event": "transfer_resume" if resume else "transfer_init",
        "checkpoint": str(latest_path if resume else tcfg["init_checkpoint"]),
        "source_epoch": source.get("epoch"), "source_val_loss": source.get("val_loss"),
        "pdb_train": len(pdb_train), "pdb_valid": len(pdb_valid),
        "drugs_train": len(drugs_train), "drugs_valid": len(drugs_valid),
    }, sort_keys=True))
    if resume:
        baseline = source.get("transfer_baseline")
        if not baseline:
            raise ValueError(f"resume checkpoint lacks fixed transfer_baseline: {latest_path}")
        initial_pdb = float(baseline["pdb_val"])
        initial_drugs = float(baseline["drugs_val"])
        log(f"fixed transfer baseline restored: pdb_val={initial_pdb:.6f} drugs_val={initial_drugs:.6f}")
    else:
        initial_pdb, initial_pdb_ratios = _evaluate_ratios(model, pdb_valid_loader, cfg, device, use_amp)
        initial_drugs, initial_drugs_ratios = _evaluate_ratios(model, drugs_valid_loader, cfg, device, use_amp)
        baseline = {
            "pdb_val": initial_pdb, "pdb_ratios": initial_pdb_ratios,
            "drugs_val": initial_drugs, "drugs_ratios": initial_drugs_ratios,
        }
        log(f"initial pdb_val={initial_pdb:.6f} ratios={initial_pdb_ratios} drugs_val={initial_drugs:.6f} ratios={initial_drugs_ratios}")

    def prior_best(name: str) -> float:
        path = checkpoint_dir / f"{run_name}_{name}.pt"
        if not resume or not path.exists():
            return float("inf")
        return float(torch.load(path, map_location="cpu", weights_only=False)["val_loss"])

    best_feasible = prior_best("best")
    best_unconstrained = prior_best("best_pdb")
    retention_tolerance = float(tcfg.get("drugs_retention_tolerance", 0.05))
    for epoch in range(start_epoch, tcfg["epochs"]):
        union.set_epoch(epoch); batch_sampler.set_epoch(epoch)
        model.train(); optimizer.zero_grad(set_to_none=True)
        weighted_loss = 0.0; graphs_seen = 0; accumulated_graphs = 0; skipped = 0
        total_microbatches = len(train_loader)
        for microbatch, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}
            batch_graphs = int(batch["node_mask"].shape[0])
            with torch.autocast(device_type="cuda", enabled=use_amp):
                losses = compute_losses(model, batch, loss_cfg)
            if not torch.isfinite(losses["loss_total"]):
                optimizer.zero_grad(set_to_none=True); accumulated_graphs = 0; skipped += 1
                continue
            scaler.scale(losses["loss_total"] * batch_graphs).backward()
            accumulated_graphs += batch_graphs
            graphs_seen += batch_graphs
            weighted_loss += float(losses["loss_total"].detach()) * batch_graphs
            should_step = accumulated_graphs >= target_graphs or microbatch == total_microbatches
            if not should_step:
                continue
            scaler.unscale_(optimizer)
            parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
            for parameter in parameters:
                parameter.grad.div_(max(accumulated_graphs, 1))
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
            stepped = False
            if torch.isfinite(grad_norm):
                scaler.step(optimizer); scaler.update(); scheduler.step(); ema.update(model); global_step += 1
                stepped = True
            else:
                scaler.update(); skipped += 1
            optimizer.zero_grad(set_to_none=True); accumulated_graphs = 0
            if stepped and global_step % tcfg.get("log_every", 100) == 0:
                log(f"epoch={epoch+1} step={global_step} lr={optimizer.param_groups[0]['lr']:.3e} train_loss={weighted_loss/max(graphs_seen,1):.5f}")

        live_state = copy.deepcopy(model.state_dict()); ema.copy_to(model)
        pdb_val, pdb_ratios = _evaluate_ratios(model, pdb_valid_loader, cfg, device, use_amp)
        drugs_val, drugs_ratios = _evaluate_ratios(model, drugs_valid_loader, cfg, device, use_amp)
        model.load_state_dict(live_state)
        retention_ok = drugs_val <= initial_drugs * (1.0 + retention_tolerance)
        log(
            f"epoch={epoch+1}/{tcfg['epochs']} train={weighted_loss/max(graphs_seen,1):.6f} "
            f"pdb_val={pdb_val:.6f} pdb_ratios={pdb_ratios} drugs_val={drugs_val:.6f} "
            f"drugs_ratios={drugs_ratios} retention_ok={retention_ok} skipped={skipped}"
        )
        _save_transfer_checkpoint(
            latest_path, model, optimizer, scheduler, ema, cfg, epoch=epoch + 1,
            val_loss=pdb_val, tag="latest", global_step=global_step, transfer_baseline=baseline,
        )
        if pdb_val < best_unconstrained:
            best_unconstrained = pdb_val
            _save_transfer_checkpoint(
                checkpoint_dir / f"{run_name}_best_pdb.pt", model, optimizer, scheduler, ema,
                cfg, epoch=epoch + 1, val_loss=pdb_val, tag="best_pdb",
                global_step=global_step, transfer_baseline=baseline,
            )
        if retention_ok and pdb_val < best_feasible:
            best_feasible = pdb_val
            _save_transfer_checkpoint(
                checkpoint_dir / f"{run_name}_best.pt", model, optimizer, scheduler, ema,
                cfg, epoch=epoch + 1, val_loss=pdb_val, tag="best_retention_feasible",
                global_step=global_step, transfer_baseline=baseline,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
