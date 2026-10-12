"""The leaves as a flat outline on the wall, from each view's green outline (hsv/segment.py).

The leaves lie flat on the wall the fruit stands on: the plane through the fruit's base
(fruit.centre) with normal fruit.normal. So no matching across views is needed:
    1. Each view's outline is undistorted and its camera rays are intersected with
       the wall plane, giving the outline in plane coordinates (world units).
    2. The outlines are filled on a raster of resolution in the plane and kept where
       at least min_views views agree (all views if fewer), so a view that sees only
       part of the star, or a strip of green clutter, does not change the shape.
    3. The largest region left, simplified to within resolution, is the leaves'
       outline, lifted back into 3D.
Each view's agreement with the result (IoU) is reported as a check on the calibration
and on the wall plane.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from tracking3d.triangulate import Camera

from ...fruit import Fruit

DEFAULT_MESH = {
    "resolution": 0.5,       # world units (mm with t_scale 1000)
    "max_extent": 200,       # half-width of the raster around the fruit (world units)
    "min_views": 2,
    "colour": [0.45, 0.68, 0.45],   # muted: bright green is ground truth in the track plots
    "alpha": 0.45,
}


@dataclass
class Leaves:
    outline: list[list[float]]          # (N, 3) world, a closed polygon on the wall
    area: float                         # world units^2
    iou: dict[str, float]               # camera -> agreement of its outline with the result

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Leaves":
        return cls(**d)


def wall_basis(fruit: Fruit) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(origin, normal, u, v): the wall plane through the fruit's base, u / v in-plane axes."""
    n = np.asarray(fruit.normal, float)
    u = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    return np.asarray(fruit.centre, float), n, u, np.cross(n, u)


def to_wall(cam: Camera, xy_undistorted: np.ndarray, fruit: Fruit) -> np.ndarray | None:
    """(N, 2) plane coordinates where the camera rays through the pixels hit the wall."""
    o, n, u, v = wall_basis(fruit)
    rays = np.c_[xy_undistorted, np.ones(len(xy_undistorted))] @ np.linalg.inv(cam.K).T @ cam.R   # R^T K^-1 p
    denom = rays @ n
    if np.any(np.abs(denom) < 1e-9):
        return None
    t = ((o - cam.centre) @ n) / denom
    if np.any(t <= 0):
        return None
    p = cam.centre + t[:, None] * rays - o
    return np.c_[p @ u, p @ v]


def reconstruct_leaves(cams: list[Camera], outlines: list[np.ndarray | None], fruit: Fruit, s: dict) -> Leaves | None:
    """outlines: per camera, the raw-pixel outline (or None). None if fewer than 2 views."""
    res, half = float(s["resolution"]), float(s["max_extent"])
    size = int(round(2 * half / res))
    names, rasters = [], []
    for cam, xy in zip(cams, outlines):
        if xy is None or len(xy) < 3:
            continue
        uv = to_wall(cam, cam.undistort(xy), fruit)
        if uv is None:
            continue
        r = np.zeros((size, size), np.uint8)
        cv2.fillPoly(r, [np.round((uv + half) / res).astype(np.int32)], 1)
        names.append(cam.name)
        rasters.append(r)
    if len(rasters) < 2:
        return None
    votes = np.sum(rasters, axis=0)
    keep = (votes >= min(int(s["min_views"]), len(rasters))).astype(np.uint8)
    contours, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 1.0, True)[:, 0].astype(float)   # 1 raster cell = resolution
    uv = c * res - half
    o, _, u, v = wall_basis(fruit)
    outline = o + uv[:, :1] * u + uv[:, 1:] * v
    region = np.zeros_like(keep)
    cv2.fillPoly(region, [c.astype(np.int32)], 1)
    iou = {n: float(np.logical_and(r, region).sum() / max(np.logical_or(r, region).sum(), 1))
           for n, r in zip(names, rasters)}
    return Leaves(outline.tolist(), float(region.sum() * res * res), iou)
