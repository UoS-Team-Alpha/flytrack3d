"""The static arena image of a view: the per-pixel median of frames spread over its recording.
Flies and other movers vanish, the arena stays. Cached in memory, so the fruit and the
leaves read each video once."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from common.datasets import IMAGE_EXTS


def median_frame(source: dict, n_frames: int) -> np.ndarray | None:
    """Per-pixel median (BGR) of n_frames frames spread over a view's video or image folder
    (a view's dataset source: "video" or "images"). None if neither can be read."""
    img = _median(source.get("video"), source.get("images"), int(n_frames))
    return None if img is None else img.copy()


@lru_cache(maxsize=8)
def _median(video: str | None, images: str | None, n_frames: int) -> np.ndarray | None:
    frames = []
    if video and Path(video).is_file():
        cap = cv2.VideoCapture(video)
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            for i in np.unique(np.linspace(0, total - 1, n_frames).astype(int)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, bgr = cap.read()
                if ok:
                    frames.append(bgr)
        finally:
            cap.release()
    elif images and Path(images).is_dir():
        imgs = sorted(p for p in Path(images).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        for i in np.unique(np.linspace(0, len(imgs) - 1, n_frames).astype(int)) if imgs else []:
            bgr = cv2.imread(str(imgs[i]), cv2.IMREAD_COLOR)
            if bgr is not None:
                frames.append(bgr)
    if not frames:
        return None
    return np.median(np.stack(frames), axis=0).astype(np.uint8)
