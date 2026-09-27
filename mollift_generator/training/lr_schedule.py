"""Learning rate schedules aligned with dit_mc optax warmup_cosine_decay."""

from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR


def build_warmup_cosine_scheduler(
    optimizer,
    *,
    total_steps: int,
    peak_lr: float,
    init_lr: float = 1e-5,
    warmup_fraction: float = 0.01,
    decay_fraction: float = 0.99,
):
    """Warmup from init_lr to peak_lr, then cosine decay to 0 over decay_fraction * total_steps."""
    warmup_steps = max(1, int(total_steps * warmup_fraction))
    decay_steps = max(warmup_steps + 1, int(total_steps * decay_fraction))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            progress = step / warmup_steps
            lr = init_lr + (peak_lr - init_lr) * progress
        elif step < decay_steps:
            progress = (step - warmup_steps) / (decay_steps - warmup_steps)
            lr = peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            lr = 0.0
        return lr / peak_lr

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
