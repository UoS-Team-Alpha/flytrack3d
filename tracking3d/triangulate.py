"""Cross-view matching and triangulation of each frame's detections (cameras: camera.py).

Per-frame matching (`match_frame`):
    1. For every pair of views, Hungarian assignment on the symmetric epipolar
       distance between detections, gated by max_epipolar_px.
    2. Pairwise matches are merged into multi-view groups, cheapest first,
       never letting a group hold two detections from the same view.
    3. Each group with >= 2 views is triangulated (linear DLT). A group whose
       reprojection error exceeds max_reprojection_px loses its worst view
       while it still has 3+; groups left with < 2 views are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .camera import Camera, epipolar_distance, fundamental, reprojection_errors, triangulate_dlt


# ----------------------------------------------------------------------------- matching

@dataclass
class Point3D:
    xyz: np.ndarray
    views: dict[int, int]                 # view index -> detection index in that view
    reproj_px: float                      # mean reprojection error over its views
    conf: float = 0.0                     # mean detection confidence


@dataclass
class MatchSettings:
    max_epipolar_px: float = 15.0
    max_reprojection_px: float = 10.0


def _group_detections(dets: list[np.ndarray], Fs: dict, gate: float) -> list[dict[int, int]]:
    """Hungarian matching per view pair, then merge into groups (one detection per view)."""
    edges = []
    for (i, j), F in Fs.items():
        if not len(dets[i]) or not len(dets[j]):
            continue
        cost = epipolar_distance(F, dets[i][:, :2], dets[j][:, :2])
        for r, c in zip(*linear_sum_assignment(cost)):
            if cost[r, c] <= gate:
                edges.append((cost[r, c], (i, r), (j, c)))

    parent: dict = {}
    members: dict = {}

    def find(n):
        while parent.setdefault(n, n) != n:
            n = parent[n]
        return n

    for _, a, b in sorted(edges):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        ma, mb = members.get(ra, {a[0]: a[1]}), members.get(rb, {b[0]: b[1]})
        if ma.keys() & mb.keys():
            continue  # would put two detections from one view in the same group
        parent[rb] = ra
        members[ra] = {**ma, **mb}
        members.pop(rb, None)
    return [members[r] for r in {find(n) for n in parent} if r in members]


def match_frame(cams: list[Camera], dets: list[np.ndarray], Fs: dict, s: MatchSettings) -> list[Point3D]:
    """dets[v]: (N, 3) undistorted x, y, conf for view v. Returns 3D points seen in >= 2 views."""
    points = []
    for group in _group_detections(dets, Fs, s.max_epipolar_px):
        while len(group) >= 2:
            vs = sorted(group)
            cs = [cams[v] for v in vs]
            xy = [dets[v][group[v], :2] for v in vs]
            X = triangulate_dlt(cs, xy)
            err = reprojection_errors(cs, xy, X)
            in_front = all(c.depth(X)[0] > 0 for c in cs)
            if in_front and err.max() <= s.max_reprojection_px:
                conf = float(np.mean([dets[v][group[v], 2] for v in vs]))
                points.append(Point3D(X, dict(group), float(err.mean()), conf))
                break
            if len(group) == 2:
                break
            group = {v: d for v, d in group.items() if v != vs[int(err.argmax())]}
    return points


def pairwise_fundamentals(cams: list[Camera]) -> dict[tuple[int, int], np.ndarray]:
    return {(i, j): fundamental(cams[i], cams[j]) for i, j in combinations(range(len(cams)), 2)}


def triangulate_session(cams: list[Camera], detections: list[dict[int, np.ndarray]],
                        s: MatchSettings) -> dict[int, list[Point3D]]:
    """detections[v] = {frame: (N, 3) undistorted x, y, conf}. Returns {frame: [Point3D]}."""
    Fs = pairwise_fundamentals(cams)
    frames = sorted(set().union(*(d.keys() for d in detections)))
    empty = np.zeros((0, 3))
    out = {}
    for f in frames:
        pts = match_frame(cams, [d.get(f, empty) for d in detections], Fs, s)
        if pts:
            out[f] = pts
    return out


def camera_baseline_report(cams: list[Camera]) -> list[str]:
    """Human-readable sanity check of the extrinsics: centres, baselines, angles between axes."""
    lines = [f"  {c.name:<14} centre ({', '.join(f'{v:9.2f}' for v in c.centre)})" for c in cams]
    for a, b in combinations(cams, 2):
        base = np.linalg.norm(a.centre - b.centre)
        ang = np.degrees(np.arccos(np.clip(a.axis @ b.axis, -1, 1)))
        lines.append(f"  {a.name} <-> {b.name}: baseline {base:.2f}, angle between axes {ang:.1f} deg")
    return lines
