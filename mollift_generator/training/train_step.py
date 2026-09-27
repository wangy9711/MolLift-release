"""Shared optimizer step with AMP, gradient clipping, and NaN guards."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    loss: torch.Tensor,
    grad_clip_norm: Optional[float] = None,
) -> bool:
    """Run backward + optimizer step. Returns False if loss/grads are non-finite."""
    if not torch.isfinite(loss).all():
        optimizer.zero_grad(set_to_none=True)
        return False

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    params = [p for p in model.parameters() if p.grad is not None]
    if params:
        grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip_norm or float("inf"))
        if not math.isfinite(float(grad_norm)):
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            return False

    scaler.step(optimizer)
    scaler.update()
    return True


def model_has_nan_params(model: nn.Module) -> bool:
    for p in model.parameters():
        if not torch.isfinite(p).all():
            return True
    return False
