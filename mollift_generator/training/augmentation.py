"""SO(3) rotation augmentation aligned with dit_mc training/utils.py."""

from __future__ import annotations

from typing import Dict

import torch


def random_rotation_matrices(batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Sample uniform random SO(3) rotations [B, 3, 3] via quaternion method."""
    u1 = torch.rand(batch_size, device=device, dtype=dtype)
    u2 = torch.rand(batch_size, device=device, dtype=dtype)
    u3 = torch.rand(batch_size, device=device, dtype=dtype)
    q1 = torch.sqrt(1 - u1) * torch.sin(2 * torch.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * torch.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2 * torch.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2 * torch.pi * u3)
    return torch.stack(
        [
            torch.stack(
                [
                    1 - 2 * (q3**2 + q4**2),
                    2 * (q2 * q3 - q1 * q4),
                    2 * (q2 * q4 + q1 * q3),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2 * (q2 * q3 + q1 * q4),
                    1 - 2 * (q2**2 + q4**2),
                    2 * (q3 * q4 - q1 * q2),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2 * (q2 * q4 - q1 * q3),
                    2 * (q3 * q4 + q1 * q2),
                    1 - 2 * (q2**2 + q3**2),
                ],
                dim=-1,
            ),
        ],
        dim=-2,
    )


def apply_rotation_augmentation(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Rotate each graph and recompute its synthetic orthographic projection.

    Projection must follow the 3D rotation: rotating planar coordinates alone
    would discard the original depth contribution.
    """
    x3d = batch["x3d"]
    b, n, _ = x3d.shape
    device = x3d.device
    dtype = x3d.dtype
    node_mask = batch["node_mask"]
    mask_f = node_mask.unsqueeze(-1).to(dtype)

    rot = random_rotation_matrices(b, device, dtype)
    x3d_rot = torch.einsum("bij,bnj->bni", rot, x3d) * mask_f

    out = dict(batch)
    out["x3d"] = x3d_rot

    if "pos_2d" in batch:
        out["pos_2d"] = x3d_rot[..., :2] * mask_f

    return out
