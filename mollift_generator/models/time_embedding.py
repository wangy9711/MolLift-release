"""Time embedding with random Fourier features."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, num_fourier: int | None = None):
        super().__init__()
        num_fourier = num_fourier or hidden_dim // 2
        # The MLP depends on this frequency draw; save it with the model weights.
        self.register_buffer(
            "freqs",
            torch.randn(num_fourier) * 2 * math.pi,
            persistent=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(num_fourier * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] -> [B, hidden_dim]"""
        if t.ndim == 0:
            t = t.unsqueeze(0)
        t = t.unsqueeze(-1)
        f = t * self.freqs.unsqueeze(0)
        ff = torch.cat([torch.sin(f), torch.cos(f)], dim=-1)
        return self.mlp(ff)
