"""Geometry shared by the arena objects: the back wall the fruit stands on, which way is up
on it, and where camera rays meet it (or any plane parallel to it).

The wall is the plane through the fruit's base (fruit.centre) with normal fruit.normal,
pointing out of the wall towards the cameras. "Up" is the cameras' mean image-up
direction (-R[1], the cameras are mounted upright) laid into the wall, so it needs no
named cameras; "right" completes a right-handed (right, up, normal) frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def perpendicular_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors perpendicular to n and to each other (any rotation about n)."""
    n = np.asarray(n, float)
    u = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


@dataclass
class Wall:
    origin: np.ndarray        # the fruit's base, on the wall
    normal: np.ndarray        # unit, out of the wall towards the cameras
    right: np.ndarray         # unit, in the wall
    up: np.ndarray            # unit, in the wall

    def to_plane(self, X: np.ndarray) -> np.ndarray:
        """(N, 3) world -> (N, 2) (right, up) coordinates, relative to the origin."""
        p = np.asarray(X, float).reshape(-1, 3) - self.origin
        return np.c_[p @ self.right, p @ self.up]

    def to_world(self, uv: np.ndarray, offset: float = 0.0) -> np.ndarray:
        """(N, 2) (right, up) coordinates -> (N, 3) world, `offset` out of the wall."""
        uv = np.asarray(uv, float).reshape(-1, 2)
        return self.origin + uv[:, :1] * self.right + uv[:, 1:] * self.up + offset * self.normal

    def hit(self, cam, xy_undistorted: np.ndarray, offset: float = 0.0) -> np.ndarray | None:
        """(N, 3) points where the camera rays through the pixels meet the plane `offset` out of
        the wall, or None if any ray is parallel to it or meets it behind the camera."""
        rays = np.c_[xy_undistorted, np.ones(len(xy_undistorted))] @ np.linalg.inv(cam.K).T @ cam.R  # R^T K^-1 p
        denom = rays @ self.normal
        if np.any(np.abs(denom) < 1e-9):
            return None
        t = ((self.origin + offset * self.normal - cam.centre) @ self.normal) / denom
        if np.any(t <= 0):
            return None
        return cam.centre + t[:, None] * rays


def wall_of(fruit, cams) -> Wall:
    """The wall the fruit stands on, oriented by the cameras' image-up directions."""
    n = np.asarray(fruit.normal, float)
    up = np.mean([-c.R[1] for c in cams], axis=0)
    up -= (up @ n) * n
    if np.linalg.norm(up) < 1e-6:   # cameras looking straight along the wall's up: any frame will do
        right, up = perpendicular_basis(n)
    else:
        up /= np.linalg.norm(up)
        right = np.cross(up, n)
    return Wall(np.asarray(fruit.centre, float), n, right, up)
