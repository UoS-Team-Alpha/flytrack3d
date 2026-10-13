"""Draw the arena frame in the 3D plots of visualise/plots3d/scene.py: its pipes as thin lines
with a faint floor and a "top" label in 3D axes, the same in the orthographic panels,
and its pipes reprojected (dashed) onto camera images."""

from __future__ import annotations

import numpy as np
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ...drawing import project_visible, style
from .reconstruct import DEFAULT_MESH, ArenaFrame

LW = 0.9
EDGE_SAMPLES = 24   # points per pipe when reprojecting, so lens distortion bends it as in the image


def draw_frame_3d(ax, frame: ArenaFrame, s: dict | None = None, label: str | None = None,
                  alpha: float | None = None) -> None:
    """alpha (if given) fades the pipes too, e.g. for the zoomed-out camera rig."""
    s = style(DEFAULT_MESH, s)
    line_alpha = 1.0 if alpha is None else min(1.0, 2 * alpha)
    for n, (a, b) in enumerate(frame.edges):
        ax.plot(*np.array([a, b]).T, color=s["colour"], lw=LW, alpha=line_alpha, label=label if n == 0 else None)
    ax.add_collection3d(Poly3DCollection([frame.floor], facecolor=s["colour"], edgecolor="none",
                                         alpha=float(s["alpha"])))
    top = (frame.front[2] + frame.front[3]) / 2 + 0.04 * np.ptp(frame.points, axis=0).max() * frame.up
    ax.text(*top, "top", color=s["colour"], fontsize=7, ha="center", alpha=line_alpha)


def draw_frame_panel(ax, frame: ArenaFrame, i: int, j: int, s: dict | None = None) -> None:
    s = style(DEFAULT_MESH, s)
    for a, b in frame.edges:
        ax.plot([a[i], b[i]], [a[j], b[j]], color=s["colour"], lw=LW, zorder=1)
    ax.add_patch(Polygon(frame.floor[:, [i, j]], closed=True, facecolor=s["colour"], alpha=float(s["alpha"]),
                         lw=0, zorder=0.5))


def draw_frame_image(ax, frame: ArenaFrame, camera, s: dict | None = None) -> None:
    """The pipes reprojected into a camera image (raw pixels, with distortion), dashed."""
    s = style(DEFAULT_MESH, s)
    t = np.linspace(0, 1, EDGE_SAMPLES)[:, None]
    for n, (a, b) in enumerate(frame.edges):
        uv = project_visible(camera, a + t * (b - a))
        if uv is not None:
            ax.plot(uv[:, 0], uv[:, 1], ls="--", color=s["colour"], lw=1.2, zorder=4,
                    label="arena frame (grown from the fruit)" if n == 0 else None)
