"""Locate the fake fruit for a session: HSV blobs in every view, triangulated once per run.

The result is cached in <run>/3d/fruit/fruit.json, next to one debug image per camera
(hsv/<camera>.jpg: red mask, candidate blobs, the blob that was used). It is recomputed
when the settings or the calibration change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..frames import median_frame
from ..settings import cache_key, object_settings, read_cache, write_cache
from .hsv.segment import DEFAULT_HSV, find_blobs, red_mask, save_debug_image
from .mesh.reconstruct import DEFAULT_MESH, Fruit, reconstruct_fruit

DEFAULT_FRUIT = {"enabled": True, "frames": 15}


def fruit_settings(config: dict) -> dict:
    """config["arena_plotting"]["fruit"] with defaults filled in."""
    return object_settings(config, "fruit", DEFAULT_FRUIT, DEFAULT_MESH, DEFAULT_HSV)


def locate_fruit(session, config: dict, refit: bool = False) -> Fruit | None:
    """The session's fruit, from the cache or freshly fitted. None (with a printed reason) if
    disabled or not found in at least two views consistently."""
    s = fruit_settings(config)
    if not s["enabled"]:
        return None
    out_dir = Path(session.run.path) / "3d" / "fruit"
    cache = out_dir / "fruit.json"
    key = cache_key(s, session.views)
    hit, saved = read_cache(cache, key)
    if hit and not refit:
        fruit = Fruit.from_json(saved) if saved else None
        print(f"\nFruit: {_describe(fruit)} (cached, {cache})")
        return fruit

    images, masks, blobs = [], [], []
    for v in session.views:
        img = median_frame(v.source, int(s["frames"]))
        mask = red_mask(img, s["hsv"]) if img is not None else None
        images.append(img)
        masks.append(mask)
        blobs.append(find_blobs(mask, s["hsv"]) if mask is not None else [])
    fruit, chosen = reconstruct_fruit([v.camera for v in session.views], blobs, s["mesh"])
    for v, img, mask, bl in zip(session.views, images, masks, blobs):
        if img is not None:
            save_debug_image(img, mask, bl, chosen.get(v.name), out_dir / "hsv" / f"{v.name}.jpg",
                             f"{v.name}: {len(bl)} candidate(s), green = used")
    write_cache(cache, key, fruit.to_json() if fruit else None)
    counts = ", ".join(f"{v.name} {len(b)}" for v, b in zip(session.views, blobs))
    print(f"\nFruit: {_describe(fruit)}\n  red candidates per view: {counts}; masks in {out_dir / 'hsv'}")
    return fruit


def _describe(fruit: Fruit | None) -> str:
    if fruit is None:
        return ("not found (no red blob triangulates consistently in >= 2 views; "
                "see arena_plotting.fruit in the config)")
    err = ", ".join(f"{k} {e:.1f}" for k, e in fruit.reprojection_px.items())
    return (f"centre {np.round(fruit.centre, 1).tolist()}, radius {fruit.radius:.1f} "
            f"(measured {fruit.radius_measured:.1f}), reprojection px: {err}")
