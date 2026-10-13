"""The arena frame, grown out from the fruit: no segmentation, only its measured size.

The arena is a tapered box: a front face and a smaller back face, joined by angled
sides, the back face on the wall the fruit stands on (arena_plotting/geometry.py).
    1. The back face lies on the wall, upright (the cameras' "up"), placed from the fruit:
       fruit_height above its bottom edge and fruit_across right of its centre line.
    2. The front face is front_size, depth out of the wall towards the cameras, on the
       same centre line (the sides taper evenly).
    3. The bottom edges are the floor: the bottom of the frame is the bottom of the arena.
It is only as accurate as the fruit's placement on the back wall; the dashed outline in
the reprojection images shows whether it sits on the real pipes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tracking3d.camera import Camera

from ...fruit import Fruit
from ...geometry import wall_of

DEFAULT_MESH = {
    "front_size": [200, 200],   # width, height between pipe centres (world units)
    "back_size": [157, 157],
    "depth": 170,               # back wall (fabric, at the fruit) to the front pipes, along the wall normal:
                                # fitted to the images (150 measured between pipes; the fabric sits further back)
    "fruit_height": 68,         # fruit centre above the back face's bottom edge
    "fruit_across": 0,          # fruit centre right of the back face's centre line
    "colour": [0.3, 0.3, 0.3],
    "alpha": 0.08,              # the floor's fill; edges are drawn solid
}

CORNERS = np.array([[-0.5, 0.0], [0.5, 0.0], [0.5, 1.0], [-0.5, 1.0]])   # bottom-left, -right, top-right, -left


@dataclass
class ArenaFrame:
    back: np.ndarray      # (4, 3) world: bottom-left, bottom-right, top-right, top-left
    front: np.ndarray     # (4, 3) world, same order
    up: np.ndarray        # (3,) unit

    @property
    def edges(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """The 12 pipes: back ring, front ring, the 4 angled side edges."""
        ring = [(i, (i + 1) % 4) for i in range(4)]
        return ([(self.back[a], self.back[b]) for a, b in ring] + [(self.front[a], self.front[b]) for a, b in ring]
                + [(self.back[i], self.front[i]) for i in range(4)])

    @property
    def floor(self) -> np.ndarray:
        """(4, 3) the floor quad: back bottom edge, front bottom edge."""
        return np.array([self.back[0], self.back[1], self.front[1], self.front[0]])

    @property
    def points(self) -> np.ndarray:
        return np.vstack([self.back, self.front])


def build_frame(fruit: Fruit, cams: list[Camera], s: dict) -> ArenaFrame:
    wall = wall_of(fruit, cams)
    (bw, bh), (fw, fh) = s["back_size"], s["front_size"]
    centre_line = -float(s["fruit_across"])                      # right coordinate of the box axis
    bottom = -float(s["fruit_height"])                           # up coordinate of the back bottom edge
    mid_height = bottom + bh / 2                                 # the box axis, at the back face's middle
    back = wall.to_world(CORNERS * [bw, bh] + [centre_line, bottom])
    front = wall.to_world(CORNERS * [fw, fh] + [centre_line, mid_height - fh / 2], offset=float(s["depth"]))
    return ArenaFrame(back, front, wall.up)
