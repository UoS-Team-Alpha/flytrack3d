"""Draw the leaves (a flat outline on the wall) in the 3D plots of tracking3d/visualise.py:
filled in 3D axes and the orthographic panels, and reprojected onto camera images."""

from __future__ import annotations

import numpy as np
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ...drawing import darker, project_visible, style
from .reconstruct import DEFAULT_MESH, Leaves


def leaves_points(leaves: Leaves) -> np.ndarray:
    return np.asarray(leaves.outline, float)


def draw_leaves_3d(ax, leaves: Leaves, s: dict | None = None, label: str | None = None,
                   alpha: float | None = None) -> None:
    s = style(DEFAULT_MESH, s)
    p = leaves_points(leaves)
    edge = darker(s["colour"])
    ax.add_collection3d(Poly3DCollection([p], facecolor=s["colour"], edgecolor="none",
                                         alpha=float(s["alpha"] if alpha is None else alpha)))
    ax.plot(*np.vstack([p, p[:1]]).T, color=edge, lw=0.8, label=label)


def draw_leaves_panel(ax, leaves: Leaves, i: int, j: int, s: dict | None = None) -> None:
    """The outline projected onto world axes (i, j). Seen near edge-on it is a thin sliver."""
    s = style(DEFAULT_MESH, s)
    xy = leaves_points(leaves)[:, [i, j]]
    ax.add_patch(Polygon(xy, closed=True, facecolor=s["colour"], edgecolor=darker(s["colour"]),
                         alpha=float(s["alpha"]), lw=0.8, zorder=1.5))
    ax.update_datalim(xy)


def draw_leaves_image(ax, leaves: Leaves, camera, s: dict | None = None) -> None:
    """Outline reprojected into a camera image (raw pixels, with distortion)."""
    s = style(DEFAULT_MESH, s)
    uv = project_visible(camera, leaves_points(leaves))
    if uv is None:
        return
    ax.add_patch(Polygon(uv, closed=True, fill=False, edgecolor=darker(s["colour"]), lw=1.4, ls="--",
                         zorder=4, label="leaves (HSV, on the wall)"))
