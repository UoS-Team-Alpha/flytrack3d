"""Visual outputs: annotated videos and frame-sized track plots.

Frames are grouped into sequences by the folder part of their key
("<sequence>/frame_000161"), so each sequence gets its own video / plot.

Neither YOLO labels nor detector outputs carry track IDs, so tracks are built by
a simple frame-to-frame linker (Hungarian matching on box-centre distance, with
a maximum distance and a maximum frame gap). This is for visual inspection only;
it is not a tracking metric.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .datasets import Frame, iter_frame_images

GT_COLOUR = (0.1, 0.8, 0.2)     # green
PRED_COLOUR = (0.9, 0.1, 0.6)   # magenta
EXTRA_COLOUR = (1.0, 0.55, 0.0)  # orange: extra points, e.g. factor-graph boxes inserted by tracking3d


def group_sequences(frames: list[Frame]) -> "OrderedDict[str, list[Frame]]":
    seqs: OrderedDict[str, list[Frame]] = OrderedDict()
    for f in frames:
        seq = f.key.rsplit("/", 1)[0] if "/" in f.key else "sequence"
        seqs.setdefault(seq, []).append(f)
    return seqs


def frame_number(frame: Frame, fallback: int) -> int:
    m = re.search(r"(\d+)$", frame.key)
    return int(m.group(1)) if m else fallback


def _safe_name(s: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", s)


# ----------------------------------------------------------------------------- tracks

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


def _centres(boxes: np.ndarray) -> np.ndarray:
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)


def _first_image(frame: Frame) -> np.ndarray | None:
    try:
        return next(iter_frame_images([frame]))[1]
    except Exception:
        return None


def save_track_plots(frames: list[Frame], preds: dict[str, np.ndarray], out_dir: Path, model_name: str,
                     conf: float, max_distance: float = 50, max_gap: int = 5,
                     background: bool = True, dpi: int = 100, extra: dict[str, np.ndarray] | None = None,
                     extra_label: str = "") -> list[Path]:
    """One frame-sized PNG per sequence with GT and predicted tracks drawn as lines.
    extra: {frame key: (N, 2) xy} points drawn as orange dots on top (legend: extra_label)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for seq, seq_frames in group_sequences(frames).items():
        w, h = seq_frames[0].width, seq_frames[0].height
        gt_pts, pr_pts = [], []
        for i, f in enumerate(seq_frames):
            fno = frame_number(f, i)
            gt_pts.append((fno, _centres(f.gt[:, 1:5])))
            p = preds.get(f.key, np.zeros((0, 6)))
            pr_pts.append((fno, _centres(p[p[:, 4] >= conf, :4])))
        gt_tracks = link_tracks(gt_pts, max_distance, max_gap)
        pr_tracks = link_tracks(pr_pts, max_distance, max_gap)

        fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        img = _first_image(seq_frames[0]) if background else None
        if img is not None:
            ax.imshow(img, extent=(0, w, h, 0), alpha=0.45)
        else:
            ax.set_facecolor("white")
        for tracks, colour, style, z in ((gt_tracks, GT_COLOUR, "-", 2), (pr_tracks, PRED_COLOUR, "--", 3)):
            for t in tracks:
                if len(t) > 1:
                    ax.plot(t[:, 1], t[:, 2], style, color=colour, lw=1.6, alpha=0.9, zorder=z)
                else:
                    ax.plot(t[:, 1], t[:, 2], "o", color=colour, ms=3, alpha=0.9, zorder=z)
        ex = [extra[f.key] for f in seq_frames if extra and f.key in extra]
        ex = np.vstack(ex) if ex else np.zeros((0, 2))
        if len(ex):
            ax.plot(ex[:, 0], ex[:, 1], "o", color=EXTRA_COLOUR, ms=3, alpha=0.9, zorder=4)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)  # image coordinates: origin top-left
        ax.set_xticks([])
        ax.set_yticks([])

        n_gt = sum(len(f.gt) for f in seq_frames)
        n_pr = sum(len(t) for t in pr_tracks)
        handles = [
            Line2D([], [], color=GT_COLOUR, ls="-", lw=2.5,
                   label=f"Ground truth ({len(gt_tracks)} tracks, {n_gt} boxes)"),
            Line2D([], [], color=PRED_COLOUR, ls="--", lw=2.5,
                   label=f"{model_name} (conf>={conf:g}; {len(pr_tracks)} tracks, {n_pr} boxes)"),
        ]
        if len(ex):
            handles.append(Line2D([], [], color=EXTRA_COLOUR, ls="", marker="o", ms=6,
                                  label=f"{extra_label} ({len(ex)} points)"))
        ax.legend(handles=handles, loc="upper left", fontsize=12, framealpha=0.85,
                  title=f"{seq}  |  frames {len(seq_frames)}", title_fontsize=11)

        path = out_dir / f"{_safe_name(seq)}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        saved.append(path)
    return saved


# ----------------------------------------------------------------------------- video

def _draw_boxes(img: np.ndarray, boxes: np.ndarray, colour_rgb, with_score: bool) -> None:
    bgr = tuple(int(255 * c) for c in colour_rgb[::-1])
    for b in boxes:
        x1, y1, x2, y2 = map(int, b[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)
        if with_score:
            cv2.putText(img, f"{b[4]:.2f}", (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1,
                        cv2.LINE_AA)


def save_videos(frames: list[Frame], preds: dict[str, np.ndarray], out_dir: Path, model_name: str,
                conf: float, fps: float = 30.0) -> list[Path]:
    """One annotated MP4 per sequence: GT boxes in green, predictions in magenta."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for seq, seq_frames in group_sequences(frames).items():
        w, h = seq_frames[0].width, seq_frames[0].height
        path = out_dir / f"{_safe_name(seq)}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f, rgb in tqdm(iter_frame_images(seq_frames), total=len(seq_frames),
                           desc=f"Video [{seq[:40]}]", unit="frame", leave=False):
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            p = preds.get(f.key, np.zeros((0, 6)))
            _draw_boxes(img, f.gt[:, 1:5], GT_COLOUR, with_score=False)
            _draw_boxes(img, p[p[:, 4] >= conf], PRED_COLOUR, with_score=True)
            cv2.rectangle(img, (0, 0), (520, 70), (0, 0, 0), -1)
            cv2.putText(img, f"GT", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        tuple(int(255 * c) for c in GT_COLOUR[::-1]), 2, cv2.LINE_AA)
            cv2.putText(img, f"{model_name} (conf>={conf:g})", (60, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        tuple(int(255 * c) for c in PRED_COLOUR[::-1]), 2, cv2.LINE_AA)
            cv2.putText(img, f.key[-45:], (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)
            writer.write(img)
        writer.release()
        saved.append(path)
    return saved
