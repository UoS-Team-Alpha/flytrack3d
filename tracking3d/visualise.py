"""3D plots: triangulated points, factor-graph tracks, ground-truth overlays and
per-camera reprojection overlays (the calibration check)."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evaluate2d.datasets import Frame, iter_frame_images  # noqa: E402
from evaluate2d.visualise import GT_COLOUR, VideoWriter, _first_image, _safe_name  # noqa: E402

# tab10 without green: green is reserved for ground truth, as in the 2D plots.
TRACK_COLOURS = tuple(c for i, c in enumerate(plt.get_cmap("tab10").colors) if i != 2)
POINTS_CMAP = "plasma"   # frame colouring for triangulated points; contains no green
# Ground truth is a wide translucent band; predictions are drawn thinner on top, so where they
# agree the prediction runs down the middle of the band and where they differ the band shows.
GT_LW, GT_ALPHA = 6.0, 0.35
PRED_POINT_SIZE = (1.5, 4.0)   # scatter size for 2-view, 3-view points
PRED_TRACK_LW = 1.2
PANELS = (("top  (X-Y)", 0, 1), ("front  (X-Z)", 0, 2))   # orthographic panels
RIG_ARENA_ALPHA = 0.25   # zoomed out to the cameras, the fruit is as small as the tracks: keep them visible
# Back-projection videos (BGR): detector boxes, ground truth, boxes the factor graph inserted.
VIDEO_DET_BGR, VIDEO_BACKPROJ_BGR = (255, 120, 0), (0, 0, 255)
VIDEO_GT_BGR = tuple(int(255 * c) for c in GT_COLOUR[::-1])


def use_interactive_backend() -> bool:
    """Switch from Agg to a GUI backend for plt.show(). False if none is available."""
    for backend in ("TkAgg", "QtAgg"):
        try:
            plt.switch_backend(backend)
            return True
        except Exception:
            continue
    return False


def _equal_3d(ax, xyz: np.ndarray) -> None:
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    centre, half = (lo + hi) / 2, max((hi - lo).max() / 2, 1e-6) * 1.05
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(centre[2] - half, centre[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def _draw_cameras(ax, cameras, length: float) -> None:
    for c in cameras:
        p, d = c.centre, c.axis * length
        ax.scatter(*p, color="black", marker="s", s=30)
        ax.quiver(*p, *d, color="black", lw=1)
        ax.text(*p, f" {c.name}", fontsize=8)


def _scene_figure(title: str, data: np.ndarray, cameras):
    """2x2 figure: 3D scene, two orthographic views, 3D rig (cameras + data).
    Returns (fig, ax3d, [ortho axes], ax_rig)."""
    fig = plt.figure(figsize=(14, 12), layout="constrained")
    fig.suptitle(title, fontsize=12)
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ortho = [fig.add_subplot(2, 2, i) for i in (2, 3)]
    rig = fig.add_subplot(2, 2, 4, projection="3d")
    _equal_3d(ax3d, data)
    rig_pts = np.vstack([data, np.array([c.centre for c in cameras])])
    _equal_3d(rig, rig_pts)
    _draw_cameras(rig, cameras, 0.15 * np.ptp(rig_pts, axis=0).max())
    ax3d.set_title("scene")
    rig.set_title("cameras + scene (calibration check)")
    # The world is the reference camera's frame, where Y points down (as image rows do). Flip a
    # panel's vertical axis where it runs against the cameras' mean image-up, so the arena's
    # floor is at the bottom (the cameras are mounted upright).
    up = np.mean([-c.R[1] for c in cameras], axis=0)
    for ax, (name, _, j) in zip(ortho, PANELS):
        ax.set_title(name)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        if up[j] < 0:
            ax.invert_yaxis()
    return fig, ax3d, ortho, rig


def _draw_gt_tracks(ax3d, ortho, rig, gt_tracks) -> None:
    """Ground-truth 3D tracks ((K, 4) frame, x, y, z) as green lines under everything else."""
    for n, t in enumerate(gt_tracks):
        label = f"ground truth ({len(gt_tracks)} tracks)" if n == 0 else None
        for ax in (ax3d, rig):
            ax.plot(*t[:, 1:4].T, color=GT_COLOUR, lw=GT_LW, alpha=GT_ALPHA, label=label if ax is ax3d else None)
        for ax, (_, i, j) in zip(ortho, PANELS):
            ax.plot(t[:, 1 + i], t[:, 1 + j], color=GT_COLOUR, lw=GT_LW, alpha=GT_ALPHA, zorder=1)


def _draw_arena(ax3d, ortho, rig, arena) -> None:
    """Static arena objects (arena_plotting.ArenaObjects: fruit, leaves), under the tracks."""
    arena.draw_3d(ax3d)
    arena.draw_3d(rig, labels=False, alpha=RIG_ARENA_ALPHA)
    for ax, (_, i, j) in zip(ortho, PANELS):
        arena.draw_panel(ax, i, j)


def plot_points_3d(points: np.ndarray, cameras, out_path: Path | None, title: str,
                   gt_tracks: list[np.ndarray] | None = None, close: bool = True,
                   arena=None):
    """points: (N, 5) frame, x, y, z, n_views. Scatter coloured by frame number; optional
    ground-truth tracks in green and arena objects (arena_plotting.ArenaObjects)."""
    gt_tracks = gt_tracks or []
    data = np.vstack([points[:, 1:4]] + [t[:, 1:4] for t in gt_tracks]
                     + ([arena.points()] if arena else []))
    fig, ax3d, ortho, rig = _scene_figure(title, data, cameras)
    _draw_gt_tracks(ax3d, ortho, rig, gt_tracks)
    if arena:
        _draw_arena(ax3d, ortho, rig, arena)
    kw = dict(c=points[:, 0], cmap=POINTS_CMAP, s=np.where(points[:, 4] >= 3, PRED_POINT_SIZE[1], PRED_POINT_SIZE[0]),
              alpha=0.9, zorder=3)
    sc = ax3d.scatter(points[:, 1], points[:, 2], points[:, 3], **kw)
    rig.scatter(points[:, 1], points[:, 2], points[:, 3], **kw)
    for ax, (_, i, j) in zip(ortho, PANELS):
        ax.scatter(points[:, 1 + i], points[:, 1 + j], **kw)
    fig.colorbar(sc, ax=[ax3d, *ortho, rig], shrink=0.5, label="frame")
    if gt_tracks or arena:
        ax3d.legend(loc="upper left", fontsize=7, framealpha=0.8)
    if out_path is not None:   # None: only build the figure (viewing a past result)
        fig.savefig(out_path, dpi=110)
    if close:
        plt.close(fig)
    return fig


def plot_tracks_3d(tracks, cameras, out_path: Path | None, title: str, points: np.ndarray | None = None,
                   gt_tracks: list[np.ndarray] | None = None,
                   anchors: list[tuple[int, str, np.ndarray]] | None = None, close: bool = True,
                   arena=None):
    """Factor-graph tracks as lines (entry = triangle, exit = square); raw triangulated points
    grey; optional ground-truth tracks in green; optional annotated points
    (track id, "start" / "end", xyz) as large plus / cross markers; optional arena objects
    (arena_plotting.ArenaObjects: fruit, leaves), included in the plot extent."""
    gt_tracks = gt_tracks or []
    data = np.vstack([t.xyz for t in tracks] + [t[:, 1:4] for t in gt_tracks])
    extent = np.vstack([data, arena.points()]) if arena else data
    fig, ax3d, ortho, rig = _scene_figure(title, extent, cameras)
    _draw_gt_tracks(ax3d, ortho, rig, gt_tracks)
    if arena:
        _draw_arena(ax3d, ortho, rig, arena)
    if points is not None and len(points):
        # Only points inside the tracks' (padded) bounding box, so clutter doesn't set the scale.
        pad = 0.1 * np.ptp(data, axis=0).max()
        inside = np.all((points[:, 1:4] >= data.min(0) - pad) & (points[:, 1:4] <= data.max(0) + pad), axis=1)
        points = points[inside]
        ax3d.scatter(points[:, 1], points[:, 2], points[:, 3], color="0.6", s=2, alpha=0.3)
        for ax, (_, i, j) in zip(ortho, PANELS):
            ax.scatter(points[:, 1 + i], points[:, 1 + j], color="0.6", s=2, alpha=0.3)
    for t in tracks:
        col = TRACK_COLOURS[(t.id - 1) % len(TRACK_COLOURS)]
        label = f"track {t.id}: frames {t.frames[0]}-{t.frames[-1]} ({t.score} dets)"
        for ax in (ax3d, rig):
            ax.plot(*t.xyz.T, color=col, lw=PRED_TRACK_LW, label=label if ax is ax3d else None)
            ax.scatter(*t.xyz[0], color=col, marker="^", s=60, edgecolor="black")
            ax.scatter(*t.xyz[-1], color=col, marker="s", s=40, edgecolor="black")
        for ax, (_, i, j) in zip(ortho, PANELS):
            ax.plot(t.xyz[:, i], t.xyz[:, j], color=col, lw=PRED_TRACK_LW, zorder=3)
            ax.plot(t.xyz[0, i], t.xyz[0, j], "^", color=col, mec="black", ms=8)
            ax.plot(t.xyz[-1, i], t.xyz[-1, j], "s", color=col, mec="black", ms=6)
    if anchors:
        _draw_anchors(ax3d, ortho, rig, anchors)
    if tracks or arena:
        ax3d.legend(loc="upper left", fontsize=7, framealpha=0.8)
    if out_path is not None:   # None: only build the figure (viewing a past result)
        fig.savefig(out_path, dpi=110)
    if close:
        plt.close(fig)
    return fig


ANCHOR_MARKERS = {"start": "P", "end": "X"}   # annotated start / end points
ANCHOR_COLOUR = "yellow"
ANCHOR_SIZE, ANCHOR_RIG_SIZE = 170 / 9, 120 / 9   # scatter sizes are areas: a third of the old width


def _draw_anchors(ax3d, ortho, rig, anchors) -> None:
    labelled = set()
    for _, kind, xyz in anchors:
        label = f"annotated {kind}" if kind not in labelled else None
        labelled.add(kind)
        kw = dict(color=ANCHOR_COLOUR, marker=ANCHOR_MARKERS[kind], edgecolor="black", linewidth=0.5, zorder=6)
        ax3d.scatter(*xyz, s=ANCHOR_SIZE, label=label, **kw)
        rig.scatter(*xyz, s=ANCHOR_RIG_SIZE, **kw)
        for ax, (_, i, j) in zip(ortho, PANELS):
            ax.scatter(xyz[i], xyz[j], s=ANCHOR_SIZE, **kw)


def save_reprojection_plots(tracks, views, out_dir: Path, dpi: int = 100,
                            annotations: dict[str, list[tuple[int, str, float, float]]] | None = None,
                            arena=None) -> list[Path]:
    """Per camera: first frame, every raw detection (grey) and the 3D tracks projected back
    through the full camera model (with distortion). Tracks that sit on the detections
    confirm the calibration; a constant offset in one view points at its extrinsics.
    Arena objects (arena_plotting.ArenaObjects), if given, are outlined dashed: they should ring
    the fruit and the leaves in the image."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for v in views:
        w, h = v.size
        fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        img = _first_image(v.background) if v.background is not None else None
        if img is not None:
            ax.imshow(img, extent=(0, w, h, 0), alpha=0.45)
        raw = np.vstack(list(v.raw.values())) if v.raw else np.zeros((0, 3))
        ax.plot(raw[:, 0], raw[:, 1], ".", color="0.35", ms=2, alpha=0.5)
        for t in tracks:
            col = TRACK_COLOURS[(t.id - 1) % len(TRACK_COLOURS)]
            front = v.camera.depth(t.xyz) > 0
            uv = v.camera.project(t.xyz[front], distort=True)
            ax.plot(uv[:, 0], uv[:, 1], "-", color=col, lw=1.6, label=f"track {t.id}")
            if len(uv):
                ax.plot(*uv[0], "^", color=col, mec="black", ms=9)
        if arena:
            arena.draw_image(ax, v.camera)
        for tid, kind, x, y in (annotations or {}).get(v.name, []):
            col = TRACK_COLOURS[(tid - 1) % len(TRACK_COLOURS)]
            ax.plot(x, y, ANCHOR_MARKERS[kind], color=col, mec="black", mew=1.2, ms=15, zorder=6)
            ax.annotate(f"#{tid} {kind}", (x, y), xytext=(10, -10), textcoords="offset points", fontsize=11,
                        color="black", bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="upper left", fontsize=11, framealpha=0.85,
                  title=f"{v.name}: grey = detections, lines = 3D tracks reprojected", title_fontsize=11)
        path = out_dir / f"{_safe_name(v.name)}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        saved.append(path)
    return saved


def _legend(img: np.ndarray, entries: list[tuple[tuple, bool, bool, str]], title: str) -> None:
    """Top-left legend on a translucent dark panel. entries: (colour, box sample, dot sample, text)."""
    font, scale, row = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 26
    width = 60 + max(cv2.getTextSize(t, font, scale, 1)[0][0] for t in [title] + [e[3] for e in entries])
    x2, y2 = min(width, img.shape[1]), min(row * (len(entries) + 1) + 12, img.shape[0])
    img[:y2, :x2] = (img[:y2, :x2] * 0.35).astype(img.dtype)
    cv2.putText(img, title, (10, row), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
    for i, (bgr, box, dot, text) in enumerate(entries, 2):
        y = row * i
        if box:
            cv2.rectangle(img, (12, y - 14), (40, y + 2), bgr, 1)
        if dot:
            cv2.circle(img, (26, y - 6), 2, bgr, -1, cv2.LINE_AA)
        cv2.putText(img, text, (50, y), font, scale, bgr, 1, cv2.LINE_AA)


def save_backproj_video(frames: list[Frame], preds: dict[str, np.ndarray], inserted: dict[str, np.ndarray],
                        track_points: dict[str, np.ndarray], path: Path, name: str, det_label: str, conf: float,
                        fps: float = 30.0, crf: int = 26) -> Path:
    """One video of a camera's frames: detector boxes (conf >= conf) in thin blue, ground truth in
    green, the factor-graph tracks reprojected as a red dot in every frame (on top of the detector
    box where there is one, to show how well the track follows it), and the boxes inserted where the
    detector has none in thin red. Boxes are (M, >=4) x1, y1, x2, y2 and track_points (N, 2) x, y,
    in raw pixels, keyed by frame key."""
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = [(VIDEO_DET_BGR, True, False, f"{det_label} (conf >= {conf:g})")]
    if any(len(f.gt) for f in frames):
        entries.append((VIDEO_GT_BGR, True, False, "ground truth"))
    entries += [(VIDEO_BACKPROJ_BGR, False, True, "factor-graph track, reprojected (every frame)"),
                (VIDEO_BACKPROJ_BGR, True, False, "factor graph, box inserted where the detector has none")]
    empty = np.zeros((0, 6))
    with VideoWriter(path, (frames[0].width, frames[0].height), fps, crf) as writer:
        for f, rgb in tqdm(iter_frame_images(frames), total=len(frames), desc=f"Video [{name}]", unit="frame",
                           leave=False):
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            p = preds.get(f.key, empty)
            ins = inserted.get(f.key, empty)
            for boxes, bgr in ((p[p[:, 4] >= conf], VIDEO_DET_BGR), (f.gt[:, 1:5], VIDEO_GT_BGR),
                               (ins, VIDEO_BACKPROJ_BGR)):
                for x1, y1, x2, y2 in np.round(boxes[:, :4]).astype(int):
                    cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 1)
            for x, y in np.round(track_points.get(f.key, np.zeros((0, 2)))).astype(int):
                cv2.circle(img, (x, y), 2, VIDEO_BACKPROJ_BGR, -1, cv2.LINE_AA)
            _legend(img, entries, f"{name}  |  {f.key.rsplit('/', 1)[-1]}")
            writer.write(img)
    return path
