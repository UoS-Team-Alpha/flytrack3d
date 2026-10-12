"""Place the arena frame for a session, grown out from its fruit (no segmentation, so no
cache: building it is instant). Needs the fruit (arena_plotting/fruit)."""

from __future__ import annotations

import numpy as np

from ..fruit import Fruit
from ..settings import object_settings
from .mesh.reconstruct import DEFAULT_MESH, ArenaFrame, build_frame

DEFAULT_ARENA = {"enabled": True}


def frame_settings(config: dict) -> dict:
    """config["arena_plotting"]["arena"] with defaults filled in."""
    return object_settings(config, "arena", DEFAULT_ARENA, DEFAULT_MESH)


def locate_frame(session, config: dict, fruit: Fruit | None) -> ArenaFrame | None:
    """The arena frame, or None (with a printed reason) if disabled or there is no fruit."""
    s = frame_settings(config)
    if not s["enabled"]:
        return None
    if fruit is None:
        print("Arena frame: skipped (it is placed from the fruit, and there is no fruit)")
        return None
    cams = [v.camera for v in session.views]
    frame = build_frame(fruit, cams, s["mesh"])
    n = np.asarray(fruit.normal)
    dist = ", ".join(f"{c.name} {(c.centre - frame.front.mean(axis=0)) @ n:.0f}" for c in cams)
    m = s["mesh"]
    print(f"Arena frame: front {m['front_size'][0]:g}x{m['front_size'][1]:g}, back {m['back_size'][0]:g}x"
          f"{m['back_size'][1]:g}, {m['depth']:g} deep, fruit {m['fruit_height']:g} above the back floor; "
          f"cameras to the front face: {dist}")
    return frame
