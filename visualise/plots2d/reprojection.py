"""Per-camera reprojection overlays: the 3D tracks projected back into each camera's image."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..common import ANCHOR_MARKERS, TRACK_COLOURS, first_image, safe_name  # noqa: E402


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
        img = first_image(v.background) if v.background is not None else None
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
        path = out_dir / f"{safe_name(v.name)}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        saved.append(path)
    return saved
