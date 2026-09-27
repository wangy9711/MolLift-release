"""Command-line entry points for training, inference and model inspection."""

from __future__ import annotations

import argparse
from importlib.resources import files
import json
import os
from pathlib import Path
import sqlite3


def registry():
    return json.loads(files("mollift").joinpath("models.json").read_text())


def checkpoint_for(args):
    if args.checkpoint:
        path = Path(args.checkpoint)
        model_name = None
    else:
        model_name = args.model or "mollift-drugs"
        entry = registry()[model_name]
        path = Path(args.checkpoint_dir) / entry["filename"]
        if path.is_file():
            from mollift.io import sha256_file
            if entry.get("sha256") and sha256_file(path) != entry["sha256"]:
                raise ValueError(f"Checksum mismatch for {model_name}: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}. See docs/models.md for placement instructions.")
    return path, model_name


def train(args):
    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    dataset = cfg["data"]["dataset"]
    cache = Path(cfg["data"]["drugs_cache_root"] if dataset == "pdb_transfer" else cfg["data"]["cache_root"])
    db = cache / "index.sqlite"
    if not db.is_file():
        raise FileNotFoundError(f"Missing {db}; follow docs/training.md first")
    if cfg["data"].get("require_geomol_split", True):
        with sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True) as con:
            try:
                protocol = con.execute("SELECT value FROM split_metadata WHERE key='split_protocol'").fetchone()
            except sqlite3.OperationalError:
                protocol = None
        if protocol != ("geomol_split0_random_1000",):
            raise ValueError("Import GeoMol split0 before training; the initial cache hash split is not the paper protocol")
    if dataset == "pdb_transfer":
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise ValueError("The released PDB transfer trainer is single-GPU; do not use torchrun")
        pdb_root = Path(cfg["data"]["pdb_root"])
        if not (pdb_root / "manifests/fragments.parquet").is_file():
            raise FileNotFoundError("PDB data are not distributed; supply a compatible private package containing manifests/fragments.parquet")
        from mollift_generator.training.train_pdb_transfer import train as execute
    elif dataset in {"geom_drugs", "geom_qm9"}:
        from mollift_generator.training.train_ddp import train_ddp as execute
    else:
        raise ValueError(f"Unsupported training dataset: {dataset}")
    execute(args.config)


def main(argv=None):
    parser = argparse.ArgumentParser(description="MolLift: molecular conformers from planar observations")
    sub = parser.add_subparsers(dest="command", required=True)
    pred = sub.add_parser("predict", help="Run G, GR, GS or GSR without a reference structure")
    weights = pred.add_mutually_exclusive_group()
    weights.add_argument("--model", choices=("mollift-drugs", "mollift-peptide"))
    weights.add_argument("--checkpoint", help="Custom MolLift checkpoint instead of a named release model")
    pred.add_argument("--checkpoint-dir", default=os.environ.get("MOLLIFT_CHECKPOINT_DIR", "checkpoints"))
    source = pred.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Calibrated angstrom JSON")
    source.add_argument("--annotation", help="MolLift annotator JSON in pixel coordinates")
    pred.add_argument("--pixels-per-angstrom", type=float, help="Override annotation bond-based scale estimation")
    pred.add_argument("--pipeline", choices=("G", "GR", "GS", "GSR"), default="GR")
    pred.add_argument("--samples", type=int, default=10)
    pred.add_argument("--steps", type=int, default=50)
    pred.add_argument("--seed", type=int, default=42)
    pred.add_argument("--batch-size", type=int, default=10, help="Generator candidate batch size; reduce for large molecules")
    pred.add_argument("--device", help="Generator device, e.g. cuda:0 or cpu (default: CUDA when available)")
    pred.add_argument("--solver-device", default="cpu")
    pred.add_argument("--solver-steps", type=int, default=600)
    pred.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads")
    pred.add_argument("--output", required=True, help="A NEW output directory")
    training = sub.add_parser("train", help="Train GEOM or fine-tune on a compatible peptide package")
    training.add_argument("--config", required=True)
    models = sub.add_parser("models", help="List model identities and optionally verify local files")
    models.add_argument("--check", action="store_true")
    models.add_argument("--checkpoint-dir", default=os.environ.get("MOLLIFT_CHECKPOINT_DIR", "checkpoints"))
    args = parser.parse_args(argv)
    try:
        if args.command == "predict":
            import torch
            if args.threads < 1:
                raise ValueError("threads must be positive")
            if args.pixels_per_angstrom is not None and not args.annotation:
                raise ValueError("pixels-per-angstrom applies only to --annotation")
            torch.set_num_threads(args.threads)
            from mollift.predict import run
            checkpoint, name = checkpoint_for(args)
            run(args, checkpoint, name)
        elif args.command == "train":
            train(args)
        else:
            entries = registry()
            if args.check:
                from mollift.io import sha256_file
                for name, entry in entries.items():
                    path = Path(args.checkpoint_dir) / entry["filename"]
                    if not path.is_file() or sha256_file(path) != entry["sha256"]:
                        raise ValueError(f"Missing or mismatched checkpoint: {name} ({path})")
            print(json.dumps(entries, indent=2))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        parser.exit(1, f"MolLift: {exc}\n")


if __name__ == "__main__":
    main()
