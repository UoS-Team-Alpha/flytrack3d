"""Draw the fake fruit in the 3D plots of tracking3d/visualise.py: as a lit surface in 3D
axes, as its filled outline in the orthographic panels, and reprojected through a camera
onto its image.

The shape stands on the wall with its base centred on fruit.centre and bulges out along
fruit.normal (towards the cameras):
    "hemisphere"  a dome of base radius fruit.radius and height fruit.depth (as in caliscope)
    "cylinder"    a disc of radius fruit.radius, fruit.depth thick
Both are a surface of revolution of a (radius, height) profile, so they share one grid.
"""

from __future__ import annotations

import numpy as np
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

from ...drawing import darker, project_visible, style
from ...geometry import perpendicular_basis
from .reconstruct import DEFAULT_MESH, Fruit

SEGMENTS = 48        # around the axis
PROFILE_STEPS = 16   # along the profile
AMBIENT = 0.45      # shading: ambient + diffuse light from the fruit's front, slightly off-axis


def _profile(shape: str) -> tuple[np.ndarray, np.ndarray]:
    """(rho, h) in units of (radius, depth), from the front (apex / face centre) to the rim on
    the wall: rho 0 -> 1, h 1 -> 0."""
    if shape == "cylinder":
        cap = np.linspace(0, 1, PROFILE_STEPS // 2)
        side = np.linspace(1, 0, PROFILE_STEPS // 2)
        return np.r_[cap, np.ones_like(side)], np.r_[np.ones_like(cap), side]
    if shape != "hemisphere":
        raise ValueError(f"arena_plotting.fruit.mesh.shape must be 'hemisphere' or 'cylinder', got '{shape}'")
    t = np.linspace(0, np.pi / 2, PROFILE_STEPS)   # angle from the apex
    return np.sin(t), np.cos(t)


def fruit_grid(fruit: Fruit, shape: str = DEFAULT_MESH["shape"]) -> np.ndarray:
    """(PROFILE_STEPS, SEGMENTS + 1, 3) surface grid; the last profile row is the rim on the wall."""
    n = np.asarray(fruit.normal, float)
    u, v = perpendicular_basis(n)
    a = np.linspace(0, 2 * np.pi, SEGMENTS + 1)
    ring = np.cos(a)[:, None] * u + np.sin(a)[:, None] * v               # (S + 1, 3)
    rho, h = _profile(shape)
    return (np.asarray(fruit.centre) + fruit.radius * rho[:, None, None] * ring
            + fruit.depth * h[:, None, None] * n)


def _shade(g: np.ndarray, fruit: Fruit, colour) -> np.ndarray:
    """(rows - 1, cols - 1, 4) facecolours lit from in front of the fruit (the cameras' side),
    off-axis so the curvature shows: ambient + Lambert."""
    n = np.asarray(fruit.normal, float)
    light = n + 0.6 * np.cross(n, [1.0, 0.0, 0.0]) + 0.3 * np.cross(n, [0.0, 1.0, 0.0])
    light /= np.linalg.norm(light)
    du, dv = g[1:, :-1] - g[:-1, :-1], g[:-1, 1:] - g[:-1, :-1]
    normals = np.cross(du, dv)
    length = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.where(length > 1e-9, normals / np.maximum(length, 1e-12), n)   # apex faces collapse
    centre = g[:-1, :-1] + (du + dv) / 2 - np.asarray(fruit.centre)   # from the base centre
    normals *= np.sign(np.sum(normals * centre, axis=-1, keepdims=True))   # point outwards
    lit = AMBIENT + (1 - AMBIENT) * np.clip(normals @ light, 0, 1)
    return np.concatenate([np.clip(np.asarray(colour)[None, None] * lit[..., None] * 1.15, 0, 1),
                           np.ones(lit.shape + (1,))], axis=-1)


def fruit_points(fruit: Fruit, s: dict | None = None) -> np.ndarray:
    """(N, 3) points spanning the fruit, for plot extents and outlines."""
    return fruit_grid(fruit, style(DEFAULT_MESH, s)["shape"]).reshape(-1, 3)


def draw_fruit_3d(ax, fruit: Fruit, s: dict | None = None, label: str | None = None,
                  alpha: float | None = None) -> None:
    """Lit surface plus a dark rim where it meets the wall. alpha overrides the configured one,
    e.g. for the zoomed-out camera rig, where the fruit would hide the tracks next to it."""
    s = style(DEFAULT_MESH, {**(s or {}), **({} if alpha is None else {"alpha": alpha})})
    g = fruit_grid(fruit, s["shape"])
    fc = _shade(g, fruit, s["colour"])
    fc[..., 3] = float(s["alpha"])
    ax.plot_surface(g[..., 0], g[..., 1], g[..., 2], facecolors=fc, shade=False, linewidth=0,
                    antialiased=False, rstride=1, cstride=1)
    rim = darker(s["colour"], 0.55)
    ax.plot(*g[-1].T, color=rim, lw=1.0, label=label)


def draw_fruit_panel(ax, fruit: Fruit, i: int, j: int, s: dict | None = None) -> None:
    """Filled outline of the fruit projected onto world axes (i, j)."""
    s = style(DEFAULT_MESH, s)
    xy = fruit_points(fruit, s)[:, [i, j]]
    hull = xy[ConvexHull(xy).vertices]
    # Background underneath, so translucent red stays red over the leaves (it covers them anyway).
    ax.add_patch(Polygon(hull, closed=True, facecolor=ax.get_facecolor(), lw=0, zorder=2))
    ax.add_patch(Polygon(hull, closed=True, facecolor=s["colour"], alpha=float(s["alpha"]), lw=0, zorder=2))
    ax.add_patch(Polygon(hull, closed=True, fill=False, edgecolor=darker(s["colour"], 0.55), lw=1.0, zorder=2))
    ax.update_datalim(hull)


def draw_fruit_image(ax, fruit: Fruit, camera, s: dict | None = None) -> None:
    """Outline of the fruit reprojected into a camera image (raw pixels, with distortion)."""
    s = style(DEFAULT_MESH, s)
    uv = project_visible(camera, fruit_points(fruit, s))
    if uv is None:
        return
    hull = uv[ConvexHull(uv).vertices]
    ax.add_patch(Polygon(hull, closed=True, fill=False, edgecolor=s["colour"], lw=1.6, ls="--", zorder=5,
                         label="fruit (HSV, triangulated)"))
