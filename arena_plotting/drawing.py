"""Small drawing helpers shared by the arena objects' mesh/draw.py modules."""

from __future__ import annotations

import numpy as np


def style(defaults: dict, s: dict | None) -> dict:
    """An object's mesh settings (colour, alpha, ...) over its defaults."""
    return {**defaults, **(s or {})}


def darker(colour, factor: float = 0.6) -> np.ndarray:
    """Edge / rim colour: the fill colour, darkened."""
    return np.asarray(colour, float) * factor


def project_visible(camera, pts: np.ndarray) -> np.ndarray | None:
    """(N, 2) raw pixels (with distortion) of world points, or None if any is behind the camera."""
    pts = np.asarray(pts, float).reshape(-1, 3)
    if np.any(camera.depth(pts) <= 0):
        return None
    return camera.project(pts, distort=True)
