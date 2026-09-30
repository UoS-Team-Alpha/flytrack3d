"""Ground-truth dataset loading.

Every dataset is flattened into a list of `Frame`s. Each frame knows how to load
its own RGB image and carries its ground truth in absolute pixel xyxy.

Supported dataset types (see config.json):
  - "yolo_images": a folder of images (optionally nested per sequence) plus a
                   mirrored YOLO labels folder. Missing label file = no objects.
  - "video":       a video file plus a folder of per-frame YOLO label files
                   named with `label_template` (e.g. frame_{:06d}).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Frame:
    key: str                      # unique id, e.g. "<sequence>/frame_000161"
    width: int
    height: int
    gt: np.ndarray                # (N, 5): class_id, x1, y1, x2, y2 in pixels
    image_path: str | None = None
    video_path: str | None = None
    video_index: int | None = None  # 0-based position in the video stream


@dataclass
class Dataset:
    name: str
    frames: list[Frame]
    class_names: dict[int, str] = field(default_factory=dict)
    source: dict = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Hash of which images are evaluated. Labels are deliberately excluded so
        editing ground truth does not invalidate cached predictions."""
        h = hashlib.sha256()
        for f in self.frames:
            h.update(f"{f.key}|{f.image_path}|{f.video_path}|{f.video_index}\n".encode())
        return h.hexdigest()[:16]


# ----------------------------------------------------------------------------- labels

def read_yolo_labels(path: Path, width: int, height: int) -> np.ndarray:
    """Read a YOLO txt file (class xc yc w h [conf]) into pixel xyxy."""
    if not path.is_file():
        return np.zeros((0, 5), dtype=np.float64)
    rows = [ln.split() for ln in path.read_text().splitlines() if ln.strip()]
    if not rows:
        return np.zeros((0, 5), dtype=np.float64)
    arr = np.array([[float(v) for v in r[:5]] for r in rows], dtype=np.float64)
    cls, xc, yc, w, h = arr.T
    return np.stack([
        cls,
        (xc - w / 2) * width,
        (yc - h / 2) * height,
        (xc + w / 2) * width,
        (yc + h / 2) * height,
    ], axis=1)


def read_class_names(data_yaml: str | Path | None) -> dict[int, str]:
    if not data_yaml:
        return {}
    data = yaml.safe_load(Path(data_yaml).read_text())
    names = data.get("names", {})
    if isinstance(names, list):
        return dict(enumerate(names))
    return {int(k): v for k, v in names.items()}


# ----------------------------------------------------------------------------- loaders

def _load_yolo_images(cfg: dict) -> list[Frame]:
    images_root = Path(cfg["images"])
    labels_root = Path(cfg["labels"])
    sequences = cfg.get("sequences")  # optional whitelist of sub-folder names

    image_paths = sorted(p for p in images_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if sequences:
        image_paths = [p for p in image_paths if p.relative_to(images_root).parts[0] in sequences]
    if not image_paths:
        raise FileNotFoundError(f"No images found under {images_root}")

    frames = []
    for p in image_paths:
        rel = p.relative_to(images_root).with_suffix("")
        with Image.open(p) as im:  # header-only read, fast
            w, h = im.size
        gt = read_yolo_labels(labels_root / rel.with_suffix(".txt"), w, h)
        frames.append(Frame(key=rel.as_posix(), width=w, height=h, gt=gt, image_path=str(p)))
    return frames


def _load_video(cfg: dict) -> list[Frame]:
    video = Path(cfg["video"])
    labels_root = Path(cfg["labels"])
    template = cfg.get("label_template", "frame_{:06d}")
    start = int(cfg.get("start_index", 1))  # label number of the first video frame
    only_labelled = bool(cfg.get("only_labelled_frames", False))

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frames = []
    for i in range(n):
        stem = template.format(i + start)
        label_path = labels_root / f"{stem}.txt"
        if only_labelled and not label_path.is_file():
            continue
        frames.append(Frame(
            key=f"{video.stem}/{stem}", width=w, height=h,
            gt=read_yolo_labels(label_path, w, h),
            video_path=str(video), video_index=i,
        ))
    return frames


LOADERS = {"yolo_images": _load_yolo_images, "video": _load_video}


def load_dataset(cfg: dict) -> Dataset:
    kind = cfg["type"]
    if kind not in LOADERS:
        raise ValueError(f"Unknown dataset type '{kind}'. Options: {list(LOADERS)}")
    frames = LOADERS[kind](cfg)
    return Dataset(
        name=cfg.get("name", kind),
        frames=frames,
        class_names=read_class_names(cfg.get("data_yaml")),
        source={k: v for k, v in cfg.items()},
    )


# ----------------------------------------------------------------------------- images

def iter_frame_images(frames: list[Frame]):
    """Yield (frame, rgb_image). Video frames are read sequentially per video."""
    caps: dict[str, cv2.VideoCapture] = {}
    positions: dict[str, int] = {}
    try:
        for f in frames:
            if f.image_path:
                bgr = cv2.imread(f.image_path, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise IOError(f"Failed to read {f.image_path}")
            else:
                cap = caps.get(f.video_path)
                if cap is None:
                    cap = caps[f.video_path] = cv2.VideoCapture(f.video_path)
                    positions[f.video_path] = 0
                if positions[f.video_path] != f.video_index:  # skip unlabelled frames
                    while positions[f.video_path] < f.video_index:
                        cap.grab()
                        positions[f.video_path] += 1
                ok, bgr = cap.read()
                positions[f.video_path] += 1
                if not ok:
                    raise IOError(f"Failed to read frame {f.video_index} of {f.video_path}")
            yield f, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        for cap in caps.values():
            cap.release()


def dataset_summary(ds: Dataset, area_ranges: dict[str, list[float]]) -> dict:
    gts = [f.gt for f in ds.frames if len(f.gt)]
    all_gt = np.concatenate(gts) if gts else np.zeros((0, 5))
    areas = (all_gt[:, 3] - all_gt[:, 1]) * (all_gt[:, 4] - all_gt[:, 2])
    sizes = sorted({(f.width, f.height) for f in ds.frames})
    return {
        "name": ds.name,
        "frames": len(ds.frames),
        "frames_with_objects": len(gts),
        "gt_boxes": int(len(all_gt)),
        "gt_boxes_by_size": {
            name: int(((areas >= lo) & (areas <= hi)).sum())
            for name, (lo, hi) in area_ranges.items() if name != "all"
        },
        "gt_boxes_by_class": {int(c): int((all_gt[:, 0] == c).sum()) for c in np.unique(all_gt[:, 0])},
        "image_sizes": [list(s) for s in sizes],
        "fingerprint": ds.fingerprint(),
        "source": json.loads(json.dumps(ds.source, default=str)),
    }
