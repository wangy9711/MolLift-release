"""Orthographic projection between 3D and a 2D plane frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import torch


@dataclass
class Plane2DFrame:
    """2D coordinate system on a plane: origin + orthonormal basis (u, v), normal n = u x v."""

    origin: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    normal: np.ndarray

    @classmethod
    def from_origin_normal_u(
        cls,
        origin: np.ndarray,
        normal: np.ndarray,
        u_axis: np.ndarray,
    ) -> "Plane2DFrame":
        origin = np.asarray(origin, dtype=float).reshape(3)
        n = np.asarray(normal, dtype=float).reshape(3)
        u = np.asarray(u_axis, dtype=float).reshape(3)

        n = n / np.linalg.norm(n)
        u = u - np.dot(u, n) * n
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-12:
            raise ValueError("u_axis is parallel to normal; cannot define in-plane x-axis")
        u = u / u_norm
        v = np.cross(n, u)
        v = v / np.linalg.norm(v)
        return cls(origin=origin, u_axis=u, v_axis=v, normal=n)

    @classmethod
    def default_xy_plane(cls) -> "Plane2DFrame":
        """XY plane: 2D (x, y) maps to world (x, y), depth along +z."""
        return cls.from_origin_normal_u(
            origin=np.zeros(3),
            normal=np.array([0.0, 0.0, 1.0]),
            u_axis=np.array([1.0, 0.0, 0.0]),
        )

    def origin_tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(self.origin, device=device, dtype=dtype)

    def u_tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(self.u_axis, device=device, dtype=dtype)

    def v_tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(self.v_axis, device=device, dtype=dtype)


def project_3d_to_2d_numpy(points_3d: np.ndarray, frame: Plane2DFrame) -> np.ndarray:
    pts = np.asarray(points_3d, dtype=float)
    rel = pts - frame.origin
    x = rel @ frame.u_axis
    y = rel @ frame.v_axis
    return np.stack([x, y], axis=-1)


def lift_2d_to_3d_numpy(points_2d: np.ndarray, frame: Plane2DFrame) -> np.ndarray:
    pts = np.asarray(points_2d, dtype=float)
    return (
        frame.origin
        + pts[:, 0:1] * frame.u_axis
        + pts[:, 1:2] * frame.v_axis
    )


def project_3d_to_2d_torch(
    coords_3d: torch.Tensor,
    frame: Plane2DFrame,
) -> torch.Tensor:
    """Orthographic projection: (N, 3) -> (N, 2)."""
    device = coords_3d.device
    dtype = coords_3d.dtype
    origin = frame.origin_tensor(device, dtype)
    u = frame.u_tensor(device, dtype)
    v = frame.v_tensor(device, dtype)
    rel = coords_3d - origin.unsqueeze(0)
    x = (rel * u.unsqueeze(0)).sum(dim=-1)
    y = (rel * v.unsqueeze(0)).sum(dim=-1)
    return torch.stack([x, y], dim=-1)


def lift_2d_to_3d_torch(
    coords_2d: torch.Tensor,
    frame: Plane2DFrame,
    depth: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """Lift 2D to 3D on the plane; optional depth along normal."""
    device = coords_2d.device
    dtype = coords_2d.dtype
    origin = frame.origin_tensor(device, dtype)
    u = frame.u_tensor(device, dtype)
    v = frame.v_tensor(device, dtype)
    n = torch.tensor(frame.normal, device=device, dtype=dtype)
    base = origin.unsqueeze(0) + coords_2d[:, 0:1] * u.unsqueeze(0) + coords_2d[:, 1:2] * v.unsqueeze(0)
    if isinstance(depth, torch.Tensor):
        return base + depth.unsqueeze(-1) * n.unsqueeze(0)
    return base + float(depth) * n.unsqueeze(0)
