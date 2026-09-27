"""Export EMA weights and inference configuration from a training checkpoint."""

import argparse
import copy
from pathlib import Path
import torch

from mollift.io import sha256_file, write_json
from mollift_generator.models.factory import build_model


def export(source: Path, output: Path, model_name: str):
    if output.exists():
        raise FileExistsError(output)
    original = torch.load(source, map_location="cpu", weights_only=False)
    state = original.get("ema_state_dict")
    if not state or "time_embed.freqs" not in state:
        raise ValueError("A complete EMA state with time_embed.freqs is required")
    source_cfg = original["config"]
    # Keep inference settings only; omit private dataset paths and optimizer state.
    cfg = {"model": copy.deepcopy(source_cfg["model"]),
           "data": {k: source_cfg["data"][k] for k in ("dataset", "max_atoms", "max_hop")},
           "train": {"prior_type": source_cfg["train"].get("prior_type", "harmonic")},
           "sample": copy.deepcopy(source_cfg["sample"])}
    model = build_model(cfg, 72, 5)
    model.load_state_dict(state, strict=True)
    payload = {"format_version": 1, "model_name": model_name, "config": cfg,
               "ema_state_dict": state, "epoch": int(original["epoch"]),
               "val_loss": float(original["val_loss"]),
               "source_checkpoint": source.name, "source_sha256": sha256_file(source),
               "weights": "EMA", "training_resume_supported": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    reloaded = torch.load(output, map_location="cpu", weights_only=True)
    assert state.keys() == reloaded["ema_state_dict"].keys()
    assert all(torch.equal(v, reloaded["ema_state_dict"][k]) for k, v in state.items())
    info = {k: payload[k] for k in ("model_name", "epoch", "val_loss", "source_checkpoint", "source_sha256", "weights")}
    info.update(filename=output.name, sha256=sha256_file(output), size_bytes=output.stat().st_size,
                dataset=cfg["data"]["dataset"], regime="mixed")
    write_json(output.with_suffix(".json"), info)
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--trusted-source", action="store_true", required=True,
                        help="Confirm the source is your trusted checkpoint (pickle loading)")
    args = parser.parse_args()
    print(export(args.source, args.output, args.model_name))
