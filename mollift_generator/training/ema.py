"""Exponential moving average of model parameters."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMAModel:
    """Track EMA shadow weights (dit_mc EMATracker analogue)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())
        for p in self.shadow.values():
            p.detach_()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        for key, shadow_val in self.shadow.items():
            if key not in model_state:
                continue
            model_val = model_state[key]
            if not torch.is_floating_point(model_val):
                self.shadow[key] = model_val.clone()
                continue
            shadow_val.mul_(self.decay).add_(model_val, alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        return copy.deepcopy(self.shadow)

    def load_state_dict(self, state_dict: dict) -> None:
        # Restore CPU-loaded tensors to the model's device and dtype before updates.
        restored = {}
        for key, value in state_dict.items():
            target = self.shadow.get(key)
            if torch.is_tensor(value) and torch.is_tensor(target):
                restored[key] = value.detach().to(device=target.device, dtype=target.dtype).clone()
            else:
                restored[key] = copy.deepcopy(value)
        self.shadow = restored

    def copy_to(self, model: nn.Module) -> None:
        """Load shadow weights into model (for eval / checkpoint)."""
        model.load_state_dict(self.state_dict(), strict=False)
