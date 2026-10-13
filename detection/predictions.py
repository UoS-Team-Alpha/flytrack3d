"""Saved predictions: the files a model writes per frame, and the cache key that lets a later
run reuse them.

A model's folder in a run holds its detections as YOLO labels in <PRED_DIR>/ and what they
were made with in <PARAMS_FILE>. The cache key covers the detector parameters (incl. weights
sha256) and the dataset fingerprint (which frames are evaluated).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from common.datasets import Dataset

from .detectors import Detector

PRED_DIR = "predictions"   # YOLO labels: <PRED_DIR>/<frame key>.txt, "class xc yc w h conf" (normalised)
PARAMS_FILE = "prediction_params.json"


def make_cache_key(detector: Detector, ds: Dataset) -> tuple[str, dict]:
    params = {"detector": detector.cache_params(), "dataset_fingerprint": ds.fingerprint()}
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16], params


def find_cached_predictions(output_root: Path, cache_key: str) -> Path | None:
    """Most recent run folder containing predictions with this cache key."""
    hits = []
    for params_path in output_root.glob(f"run_*/**/{PARAMS_FILE}"):  # single and batch layouts
        try:
            if json.loads(params_path.read_text()).get("cache_key") == cache_key \
                    and (params_path.parent / PRED_DIR).is_dir():
                hits.append(params_path.parent)
        except (json.JSONDecodeError, OSError):
            continue
    # Run folder names sort chronologically, so the newest run wins.
    return max(hits, key=lambda p: p.relative_to(output_root).parts[0]) if hits else None


def save_predictions(pred_dir: Path, ds: Dataset, preds: dict[str, np.ndarray], class_names: dict) -> None:
    """One YOLO .txt per frame (empty file = evaluated, no detections) plus a data.yaml."""
    for f in ds.frames:
        p = preds.get(f.key, np.zeros((0, 6)))
        path = pred_dir / f"{f.key}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        x1, y1, x2, y2, score, cls = p.T
        rows = zip(cls, (x1 + x2) / 2 / f.width, (y1 + y2) / 2 / f.height,
                   (x2 - x1) / f.width, (y2 - y1) / f.height, score)
        path.write_text("".join(f"{int(c)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {sc:.5f}\n"
                                for c, xc, yc, w, h, sc in rows))
    names = "".join(f"  {k}: {v}\n" for k, v in sorted(class_names.items()))
    (pred_dir / "data.yaml").write_text(f"names:\n{names}" if names else "names: {}\n")


def load_predictions(pred_dir: Path, ds: Dataset) -> dict[str, np.ndarray]:
    """Read YOLO prediction files back to {key: (M, 6) x1, y1, x2, y2, score, class} in pixels."""
    preds = {}
    for f in ds.frames:
        path = pred_dir / f"{f.key}.txt"
        rows = [ln.split() for ln in path.read_text().splitlines() if ln.strip()] if path.is_file() else []
        a = np.array(rows, dtype=np.float64).reshape(-1, 6)
        cls, xc, yc, w, h, score = a.T
        preds[f.key] = np.stack([(xc - w / 2) * f.width, (yc - h / 2) * f.height,
                                 (xc + w / 2) * f.width, (yc + h / 2) * f.height, score, cls], axis=1)
    return preds
