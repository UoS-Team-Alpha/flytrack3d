"""Triangulate the fake fruit from the red blobs found in each view (hsv/segment.py).

    1. Every combination of one candidate blob per view (a view may also sit out)
       is triangulated (linear DLT on undistorted centroids). The combination with
       the most views whose worst reprojection error is within max_reprojection_px
       wins, lowest error first. Red clutter seen by one camera does not triangulate
       consistently, so it loses to the fruit.
    2. Each blob is the fruit's outline seen from the front (its rim on the wall), so the
       triangulated point is the centre of its base, on the wall. Its radius
       is the median over views of blob radius (px) x depth / focal length, unless
       fixed in the config ("radius", world units).
    3. Its axis points out of the wall towards the cameras: the mean direction
       from the fruit to the camera centres, unless fixed ("normal").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations, product

import numpy as np

from tracking3d.triangulate import Camera, reprojection_errors, triangulate_dlt

from ..hsv.segment import Blob

DEFAULT_MESH = {
    "max_reprojection_px": 15.0,
    "radius": None,           # world units; None = measured from the blobs
    "depth": None,            # how far the fruit stands off the wall (world units); None = its radius
    "normal": None,           # [x, y, z] out of the wall; None = towards the mean camera centre
    "shape": "hemisphere",    # or "cylinder" (mesh/draw.py)
    "colour": [0.85, 0.1, 0.1],
    "alpha": 0.45,
}


@dataclass
class Fruit:
    centre: list[float]               # world, centre of its base on the wall
    normal: list[float]               # unit, out of the wall
    radius: float
    depth: float
    radius_measured: float
    pixels: dict[str, list[float]]    # camera -> raw pixel centroid used
    reprojection_px: dict[str, float]

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Fruit":
        return cls(**d)


def _best_combination(cams: list[Camera], blobs: list[list[Blob]], gate: float):
    """(view indices, blobs, X, errors) of the best-supported consistent combination, or None."""
    best = None   # sort key: more views first, then lower worst error
    for k in range(len(cams), 1, -1):
        for idx in combinations(range(len(cams)), k):
            for combo in product(*(blobs[i] for i in idx)):
                cs = [cams[i] for i in idx]
                xy = [c.undistort(b.xy[None])[0] for c, b in zip(cs, combo)]
                X = triangulate_dlt(cs, xy)
                if np.any([c.depth(X)[0] <= 0 for c in cs]):
                    continue
                err = reprojection_errors(cs, xy, X)
                if err.max() <= gate and (best is None or err.max() < best[3].max()):
                    best = (idx, combo, X, err)
        if best is not None:
            return best
    return None


def reconstruct_fruit(cams: list[Camera], blobs: list[list[Blob]], s: dict) -> tuple[Fruit | None, dict[str, Blob]]:
    """The fruit, and the blob chosen in each view that saw it (camera name -> blob)."""
    found = _best_combination(cams, blobs, float(s["max_reprojection_px"]))
    if found is None:
        return None, {}
    idx, combo, X, err = found
    used = [cams[i] for i in idx]
    radii = [b.radius_px * c.depth(X)[0] / c.K[0, 0] for c, b in zip(used, combo)]
    measured = float(np.median(radii))
    radius = float(s["radius"]) if s.get("radius") else measured
    if s.get("normal"):
        normal = np.asarray(s["normal"], float)
    else:
        normal = np.mean([(c.centre - X) / np.linalg.norm(c.centre - X) for c in cams], axis=0)
    normal /= np.linalg.norm(normal)
    fruit = Fruit(
        centre=X.tolist(), normal=normal.tolist(), radius=radius,
        depth=float(s["depth"]) if s.get("depth") else radius, radius_measured=measured,
        pixels={c.name: b.xy.tolist() for c, b in zip(used, combo)},
        reprojection_px={c.name: float(e) for c, e in zip(used, err)},
    )
    return fruit, {c.name: b for c, b in zip(used, combo)}
