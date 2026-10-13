"""Ground-truth dataset loading.

Every dataset is flattened into a list of `Frame`s. Each frame knows how to load
its own RGB image and carries its ground truth in absolute pixel xyxy.

Supported dataset types (see config.json):
  - "yolo_images": a folder of images (optionally nested per sequence) plus a
                   mirrored YOLO labels folder. Missing label file = no objects.
  - "video":       a video file plus a folder of per-frame YOLO label files
                   named with `label_template` (e.g. frame_{:06d}).
  - "batch":       a folder of inputs (video files and/or folders of frames)
                   plus a folder of label folders. Each input is paired with
                   the label folder whose name matches (or contains) its name
                   and evaluated as its own dataset - see `discover_batch`.

"labels" may point straight at the folder of .txt files, or at an export root
that contains it (e.g. a CVAT YOLO export: <root>/labels/train/*.txt). If no
"data_yaml" is given, a data.yaml in that root is used for class names. Without
"labels" (inference only, see main_inference2d.py) every frame has no ground truth.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

from .yolo_format import read_yolo_labels

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv"}
SPLIT_LISTS = {"train.txt", "val.txt", "valid.txt", "test.txt"}  # image lists, not labels


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

def _has_label_files(d: Path) -> bool:
    return any(p.suffix == ".txt" and p.name not in SPLIT_LISTS for p in d.iterdir() if p.is_file())


def resolve_label_dir(path: str | Path) -> Path:
    """Find the folder of label .txt files under `path`. Accepts the folder itself
    or an export root holding exactly one labels/<split>/ folder."""
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Labels folder not found: {root}")
    if (root / "labels").is_dir():
        found = sorted({p.parent for p in (root / "labels").rglob("*.txt")})
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise ValueError(f"Several label folders under {root / 'labels'}: "
                             f"{[str(f.relative_to(root)) for f in found]}. Point 'labels' at one of them.")
    return root


def find_data_yaml(label_root: str | Path) -> Path | None:
    """data.yaml in the given labels folder or its export root, if any."""
    p = Path(label_root)
    for d in (p, *p.parents[:2]):
        if (d / "data.yaml").is_file():
            return d / "data.yaml"
    return None


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
    labels_root = resolve_label_dir(cfg["labels"]) if cfg.get("labels") else None
    sequences = cfg.get("sequences")  # optional whitelist of sub-folder names
    prefix = cfg.get("key_prefix")    # set in batch mode so keys are "<input name>/<frame>"

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
        gt = np.zeros((0, 5))
        if labels_root:
            label_path = labels_root / rel.with_suffix(".txt")
            if not label_path.is_file():  # nested images (e.g. images/train/) with a flat labels folder
                label_path = labels_root / f"{p.stem}.txt"
            gt = read_yolo_labels(label_path, w, h)
        key = f"{prefix}/{rel.as_posix()}" if prefix else rel.as_posix()
        frames.append(Frame(key=key, width=w, height=h, gt=gt, image_path=str(p)))
    return frames


def _load_video(cfg: dict) -> list[Frame]:
    video = Path(cfg["video"])
    labels_root = resolve_label_dir(cfg["labels"]) if cfg.get("labels") else None
    template = cfg.get("label_template", "frame_{:06d}")
    start = int(cfg.get("start_index", 1))  # label number of the first video frame
    only_labelled = bool(cfg.get("only_labelled_frames", False)) and labels_root is not None

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frames, stems = [], set()
    for i in range(n):
        stem = template.format(i + start)
        stems.add(stem)
        label_path = labels_root / f"{stem}.txt" if labels_root else None
        if only_labelled and not label_path.is_file():
            continue
        frames.append(Frame(
            key=f"{video.stem}/{stem}", width=w, height=h,
            gt=read_yolo_labels(label_path, w, h) if label_path else np.zeros((0, 5)),
            video_path=str(video), video_index=i,
        ))

    if labels_root is None:
        return frames
    label_stems = {p.stem for p in labels_root.glob("*.txt") if p.name not in SPLIT_LISTS}
    if label_stems and not label_stems & stems:
        raise ValueError(f"No label files in {labels_root} match video frames "
                         f"{template.format(start)}..{template.format(start + n - 1)}. "
                         f"Check 'label_template' and 'start_index'.")
    if unmatched := sorted(label_stems - stems):
        tqdm.write(f"  WARNING: {len(unmatched)} label files in {labels_root} have no video frame "
                   f"(e.g. {unmatched[0]}; video has {n} frames). Check 'start_index' ({start}).")
    return frames


LOADERS = {"yolo_images": _load_yolo_images, "video": _load_video}


def load_dataset(cfg: dict) -> Dataset:
    kind = cfg["type"]
    if kind not in LOADERS:
        raise ValueError(f"Unknown dataset type '{kind}'. Options: {list(LOADERS) + ['batch']}")
    frames = LOADERS[kind](cfg)
    data_yaml = cfg.get("data_yaml") or (find_data_yaml(resolve_label_dir(cfg["labels"])) if cfg.get("labels") else None)
    return Dataset(
        name=cfg.get("name", kind),
        frames=frames,
        class_names=read_class_names(data_yaml),
        source={k: v for k, v in cfg.items()},
    )


# ----------------------------------------------------------------------------- batch

def _match_label_dir(name: str, label_dirs: list[Path]) -> tuple[Path | None, list[Path]]:
    """Exact name match, else the unique label folder whose name contains the input
    name (or vice versa). Returns (match, candidates)."""
    exact = [d for d in label_dirs if d.name == name]
    if exact:
        return exact[0], exact
    cands = [d for d in label_dirs if name in d.name or d.name in name]
    return (cands[0] if len(cands) == 1 else None), cands


def discover_inputs(cfg: dict) -> list[dict]:
    """Every input (video file or folder of frames) in cfg["inputs"], as a per-input dataset
    config without labels. Per-input configs inherit the batch-level settings
    (label_template, start_index, ...); cfg["items"] optionally whitelists input names."""
    inputs_root = Path(cfg["inputs"])
    if not inputs_root.is_dir():
        raise FileNotFoundError(f"Batch folder not found: {inputs_root}")
    only = set(cfg.get("items") or [])
    inputs = sorted(p for p in inputs_root.iterdir()
                    if (p.is_file() and p.suffix.lower() in VIDEO_EXTS) or p.is_dir())
    shared = {k: v for k, v in cfg.items() if k not in {"type", "name", "inputs", "labels", "items"}}

    out, names = [], set()
    for item in inputs:
        name = item.stem if item.is_file() else item.name
        if only and name not in only:
            continue
        if name in names:
            raise ValueError(f"Two inputs are both named '{name}' in {inputs_root}")
        names.add(name)
        if item.is_file():
            out.append({**shared, "type": "video", "video": str(item), "name": name})
        else:
            out.append({**shared, "type": "yolo_images", "images": str(item), "key_prefix": name, "name": name})
    return out


def discover_batch(cfg: dict) -> tuple[list[dict], list[str]]:
    """Pair every input in cfg["inputs"] (see discover_inputs) with a label folder in
    cfg["labels"]. Returns (per-pair dataset configs, warnings); each config can go
    straight to load_dataset."""
    inputs = discover_inputs(cfg)
    labels_root = Path(cfg["labels"])
    if not labels_root.is_dir():
        raise FileNotFoundError(f"Batch folder not found: {labels_root}")
    label_dirs = sorted(p for p in labels_root.iterdir() if p.is_dir())

    pairs, warnings, used = [], [], set()
    for pair in inputs:
        label_dir, cands = _match_label_dir(pair["name"], label_dirs)
        if label_dir is None:
            why = "no label folder" if not cands else f"ambiguous label folders {[c.name for c in cands]}"
            warnings.append(f"Skipping {Path(pair.get('video') or pair['images']).name}: {why}")
            continue
        used.add(label_dir)
        pairs.append({**pair, "labels": str(label_dir)})
    for d in label_dirs:
        if d not in used and not cfg.get("items"):
            warnings.append(f"Label folder {d.name} has no matching input")
    if not pairs:
        raise FileNotFoundError(f"No input/label pairs found between {cfg['inputs']} and {labels_root}")
    return pairs, warnings


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


# ----------------------------------------------------------------------------- sequences

def group_sequences(frames: list[Frame]) -> "OrderedDict[str, list[Frame]]":
    seqs: OrderedDict[str, list[Frame]] = OrderedDict()
    for f in frames:
        seq = f.key.rsplit("/", 1)[0] if "/" in f.key else "sequence"
        seqs.setdefault(seq, []).append(f)
    return seqs


def frame_number(frame: Frame, fallback: int) -> int:
    m = re.search(r"(\d+)$", frame.key)
    return int(m.group(1)) if m else fallback
