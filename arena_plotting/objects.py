"""Every static arena object of a session, located once and drawn together, so the plots in
tracking3d/visualise.py take a single `arena` argument however many objects there are."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fruit import (Fruit, draw_fruit_3d, draw_fruit_image, draw_fruit_panel, fruit_points, fruit_settings,
                    locate_fruit)
from .leaves import (Leaves, draw_leaves_3d, draw_leaves_image, draw_leaves_panel, leaves_points,
                     leaves_settings, locate_leaves)


@dataclass
class ArenaObjects:
    fruit: Fruit | None = None
    leaves: Leaves | None = None
    fruit_style: dict = field(default_factory=dict)
    leaves_style: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.fruit is not None or self.leaves is not None

    def points(self) -> np.ndarray:
        """(N, 3) points spanning every object, for plot extents."""
        parts = ([fruit_points(self.fruit, self.fruit_style)] if self.fruit else []) + \
                ([leaves_points(self.leaves)] if self.leaves else [])
        return np.vstack(parts) if parts else np.zeros((0, 3))

    def draw_3d(self, ax, labels: bool = True, alpha: float | None = None) -> None:
        """alpha overrides every object's own, e.g. faint objects in a zoomed-out view."""
        if self.leaves:
            draw_leaves_3d(ax, self.leaves, self.leaves_style, "leaves" if labels else None, alpha)
        if self.fruit:
            draw_fruit_3d(ax, self.fruit, self.fruit_style, "fruit" if labels else None, alpha)

    def draw_panel(self, ax, i: int, j: int) -> None:
        if self.leaves:
            draw_leaves_panel(ax, self.leaves, i, j, self.leaves_style)
        if self.fruit:
            draw_fruit_panel(ax, self.fruit, i, j, self.fruit_style)

    def draw_image(self, ax, camera) -> None:
        if self.leaves:
            draw_leaves_image(ax, self.leaves, camera, self.leaves_style)
        if self.fruit:
            draw_fruit_image(ax, self.fruit, camera, self.fruit_style)


def locate_arena(session, config: dict, refit: bool = False) -> ArenaObjects:
    """Fruit, then the leaves on its wall; each one is skipped if disabled or not found."""
    fruit = locate_fruit(session, config, refit)
    leaves = locate_leaves(session, config, fruit, refit)
    return ArenaObjects(fruit, leaves, fruit_settings(config)["mesh"], leaves_settings(config)["mesh"])
