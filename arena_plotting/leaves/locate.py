"""Locate the leaves for a session: the green outline around the fruit in every view, laid
onto the wall the fruit stands on. Needs the fruit (arena_plotting/fruit) for the wall plane.

The result is cached in <run>/3d/leaves/leaves.json, next to one debug image per camera
(hsv/<camera>.jpg: green mask, outline used, fruit centre). It is recomputed when the
settings, the calibration or the fruit change. The median frames are the fruit's ones
(config arena_plotting.fruit.frames).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..frames import median_frame
from ..fruit import Fruit, fruit_settings
from ..settings import cache_key, object_settings, read_cache, write_cache
from .hsv.segment import DEFAULT_HSV, green_mask, leaves_contour, save_debug_image
from .mesh.reconstruct import DEFAULT_MESH, Leaves, reconstruct_leaves

DEFAULT_LEAVES = {"enabled": True}


def leaves_settings(config: dict) -> dict:
    """config["arena_plotting"]["leaves"] with defaults filled in."""
    return object_settings(config, "leaves", DEFAULT_LEAVES, DEFAULT_MESH, DEFAULT_HSV)


def locate_leaves(session, config: dict, fruit: Fruit | None, refit: bool = False) -> Leaves | None:
    """The session's leaves, from the cache or freshly fitted. None (with a printed reason) if
    disabled, without a fruit, or outlined in fewer than two views."""
    s = leaves_settings(config)
    if not s["enabled"]:
        return None
    if fruit is None:
        print("\nLeaves: skipped (they are placed on the fruit's wall, and there is no fruit)")
        return None
    frames = int(fruit_settings(config)["frames"])
    out_dir = Path(session.run.path) / "3d" / "leaves"
    cache = out_dir / "leaves.json"
    key = cache_key(s, session.views, frames=frames, fruit={"centre": fruit.centre, "normal": fruit.normal})
    hit, saved = read_cache(cache, key)
    if hit and not refit:
        leaves = Leaves.from_json(saved) if saved else None
        print(f"Leaves: {_describe(leaves)} (cached, {cache})")
        return leaves

    outlines = []
    for v in session.views:
        img = median_frame(v.source, frames)
        if img is None:
            outlines.append(None)
            continue
        fruit_xy = v.camera.project(np.asarray(fruit.centre)[None], distort=True)[0]
        mask = green_mask(img, s["hsv"])
        outline = leaves_contour(mask, fruit_xy, s["hsv"])
        outlines.append(outline)
        save_debug_image(img, mask, outline, fruit_xy, out_dir / "hsv" / f"{v.name}.jpg",
                         f"{v.name}: {'leaves outlined' if outline is not None else 'no green region at the fruit'}")
    leaves = reconstruct_leaves([v.camera for v in session.views], outlines, fruit, s["mesh"])
    write_cache(cache, key, leaves.to_json() if leaves else None)
    print(f"Leaves: {_describe(leaves)}\n  masks in {out_dir / 'hsv'}")
    return leaves


def _describe(leaves: Leaves | None) -> str:
    if leaves is None:
        return "not found (no green region around the fruit in >= 2 views; see arena_plotting.leaves in the config)"
    p = np.asarray(leaves.outline)
    span = np.ptp(p, axis=0).max()
    iou = ", ".join(f"{k} {v:.2f}" for k, v in leaves.iou.items())
    return f"{len(p)}-point outline, {span:.0f} across, area {leaves.area:.0f}; agreement (IoU) per view: {iou}"
