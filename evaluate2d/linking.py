"""Link per-frame points into tracks.

Neither YOLO labels nor detector outputs carry track IDs, so tracks are built by a simple
frame-to-frame linker (Hungarian matching on distance, with a maximum distance and a maximum
frame gap). It works on points of any dimension: box centres for the 2D track plots, and
triangulated 3D points for tracking3d's tracklets and ground-truth overlay. On its own this is
for visual inspection and seeding only; it is not a tracking metric.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def link_tracks(points: list[tuple[int, np.ndarray]], max_distance: float, max_gap: int) -> list[np.ndarray]:
    """Link per-frame points [(frame_no, (N,2) xy), ...] into tracks.
    Returns a list of (K, 3) arrays of frame_no, x, y."""
    finished, active = [], []  # active: list of [last_frame, [(f, x, y), ...]]
    for fno, xy in points:
        still_active = []
        for t in active:
            (still_active if fno - t[0] <= max_gap else finished).append(t)
        active = still_active

        assigned = set()
        if active and len(xy):
            last = np.array([t[1][-1][1:] for t in active])
            cost = np.linalg.norm(last[:, None, :] - xy[None, :, :], axis=2)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] <= max_distance:
                    active[r][0] = fno
                    active[r][1].append((fno, *xy[c]))
                    assigned.add(c)
        for c in range(len(xy)):
            if c not in assigned:
                active.append([fno, [(fno, *xy[c])]])
    finished.extend(active)
    return [np.array(t[1]) for t in finished]
