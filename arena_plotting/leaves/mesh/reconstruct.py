"""The leaves as a flat outline on the wall, from each view's green outline (hsv/segment.py).

The leaves lie flat on the wall the fruit stands on: the plane through the fruit's base
(fruit.centre) with normal fruit.normal. So no matching across views is needed:
    1. Each view's outline is undistorted and its camera rays are intersected with
       the wall plane (arena_plotting/geometry.py), giving it in wall coordinates.
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

from tracking3d.camera import Camera

from ...fruit import Fruit
from ...geometry import wall_of

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


def reconstruct_leaves(cams: list[Camera], outlines: list[np.ndarray | None], fruit: Fruit, s: dict) -> Leaves | None:
    """outlines: per camera, the raw-pixel outline (or None). None if fewer than 2 views."""
    wall = wall_of(fruit, cams)
    res, half = float(s["resolution"]), float(s["max_extent"])
    size = int(round(2 * half / res))
    names, rasters = [], []
    for cam, xy in zip(cams, outlines):
        if xy is None or len(xy) < 3:
            continue
        X = wall.hit(cam, cam.undistort(xy))
        if X is None:
            continue
        uv = wall.to_plane(X)
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
    outline = wall.to_world(c * res - half)
    region = np.zeros_like(keep)
    cv2.fillPoly(region, [c.astype(np.int32)], 1)
    iou = {n: float(np.logical_and(r, region).sum() / max(np.logical_or(r, region).sum(), 1))
           for n, r in zip(names, rasters)}
    return Leaves(outline.tolist(), float(region.sum() * res * res), iou)
