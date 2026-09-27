"""Strict EMA checkpoint loading, including persistent Fourier frequencies."""

import torch

from mollift_generator.models.factory import build_model


def load_model_from_checkpoint(checkpoint_path, device, atom_dim=None, bond_dim=None):
    # Public exports contain tensors and ordinary dictionaries only.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    state = ckpt.get("ema_state_dict") or ckpt["model_state_dict"]
    if "time_embed.freqs" not in state:
        raise ValueError("Checkpoint lacks persistent time_embed.freqs; cannot reproduce this model")
    inferred_atom = state["graph_encoder.node_in.weight"].shape[1]
    inferred_bond = state["graph_encoder.edge_mlps.0.0.weight"].shape[1] - 2 * cfg["model"]["hidden_dim"]
    if atom_dim is not None and atom_dim != inferred_atom:
        raise ValueError("Input atom feature width differs from checkpoint")
    if bond_dim is not None and bond_dim != inferred_bond:
        raise ValueError("Input bond feature width differs from checkpoint")
    model = build_model(cfg, inferred_atom, inferred_bond)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), cfg
