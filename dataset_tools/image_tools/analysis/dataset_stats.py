"""Statistics of a YOLO dataset, per split: which videos it holds, how many images and
instances each gives, and how large the boxes are and where they sit.

The images of a split are grouped by the video they came from. A frame's video is the part
of its file name before `prefix_separator` ("<video>_frame_000123.png" -> "<video>"); frames
named without it take the name of their sub-folder of the split (images/test/<video>/...).

An image without a label file counts as a background image, as it does in training, and is
also reported as missing its label file. The text report and the charts are made from these
statistics by visualise.metrics.dataset and visualise.charts.dataset.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from common.yolo_format import find_splits, list_images, read_yolo_boxes, split_dirs

NO_VIDEO = "(no video prefix)"
SIZE_BINS = (0, 8, 16, 32, 96, np.inf)   # box size = sqrt(width * height) in px; 32 and 96 are COCO's small / medium limits


@dataclass
class VideoStats:
    images: int = 0
    background: int = 0                               # images without an object
    instances: Counter = field(default_factory=Counter)   # class id -> boxes


@dataclass
class SplitStats:
    name: str
    image_dir: Path
    label_dir: Path
    videos: dict[str, VideoStats] = field(default_factory=dict)   # in order of first appearance
    image_sizes: Counter = field(default_factory=Counter)         # (width, height) -> images
    boxes_per_image: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    # One row per box: class, x_centre, y_centre (normalised), width, height (px), index of its video in `videos`
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    missing_labels: int = 0     # images without a label file
    orphan_labels: int = 0      # label files without an image
    unreadable_images: int = 0
    out_of_bounds: int = 0      # boxes reaching outside the image

    @property
    def images(self) -> int:
        return sum(v.images for v in self.videos.values())

    @property
    def background(self) -> int:
        return sum(v.background for v in self.videos.values())

    @property
    def instances(self) -> Counter:
        return sum((v.instances for v in self.videos.values()), Counter())

    def box_sizes(self, video: str | None = None) -> np.ndarray:
        """sqrt(width * height) of every box in px, or of one video's boxes."""
        boxes = self.boxes
        if video is not None:
            boxes = boxes[boxes[:, 5] == list(self.videos).index(video)]
        return np.sqrt(boxes[:, 3] * boxes[:, 4])


def video_prefix(image_path: Path, image_dir: Path, separator: str = "_frame_") -> str:
    """The video an image came from: its name up to the separator, else its sub-folder of the split."""
    name = image_path.name
    if separator and separator in name and not name.startswith(separator):
        return name.split(separator)[0]
    folders = image_path.relative_to(image_dir).parts[:-1]
    return folders[0] if folders else NO_VIDEO


def analyse_split(dataset_dir: str | Path, split: str, prefix_separator: str = "_frame_") -> SplitStats:
    image_dir, label_dir = split_dirs(dataset_dir, split)
    stats = SplitStats(split, image_dir, label_dir)
    image_paths = list_images(image_dir, recursive=True)
    video_index: dict[str, int] = {}
    boxes, boxes_per_image = [], []

    for image_path in tqdm(image_paths, desc=f"Analysing {split}", unit="img"):
        video = video_prefix(image_path, image_dir, prefix_separator)
        v = stats.videos.setdefault(video, VideoStats())
        idx = video_index.setdefault(video, len(video_index))
        v.images += 1

        label_path = label_dir / image_path.relative_to(image_dir).with_suffix(".txt")
        if not label_path.exists():
            stats.missing_labels += 1
        rows = read_yolo_boxes(label_path)
        boxes_per_image.append(len(rows))
        if not rows:
            v.background += 1
        try:
            with Image.open(image_path) as im:   # header-only read, fast
                img_w, img_h = im.size
        except OSError:
            stats.unreadable_images += 1
            img_w = img_h = None
        else:
            stats.image_sizes[(img_w, img_h)] += 1
        for cls, x_c, y_c, w, h in rows:
            v.instances[int(cls)] += 1
            if min(x_c - w / 2, y_c - h / 2) < -1e-6 or max(x_c + w / 2, y_c + h / 2) > 1 + 1e-6:
                stats.out_of_bounds += 1
            if img_w:   # the box sizes are in px, so a box of an unreadable image is only counted
                boxes.append((cls, x_c, y_c, w * img_w, h * img_h, idx))

    stats.boxes_per_image = np.array(boxes_per_image, dtype=int)
    stats.boxes = np.array(boxes, dtype=float).reshape(-1, 6)
    if label_dir.is_dir():
        with_image = {p.relative_to(image_dir).with_suffix("") for p in image_paths}
        stats.orphan_labels = sum(p.relative_to(label_dir).with_suffix("") not in with_image
                                  for p in label_dir.rglob("*.txt"))
    return stats


def analyse_dataset(dataset_dir: str | Path, splits: list[str] | None = None,
                    prefix_separator: str = "_frame_") -> list[SplitStats]:
    """The statistics of each split; splits = None takes every split the dataset has (train, val, test)."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset folder not found: {dataset_dir}")
    splits = list(splits) if splits else find_splits(dataset_dir)
    if not splits:
        raise SystemExit(f"No split folders (images/<split> or <split>/images) found in {dataset_dir}")
    stats = []
    for split in splits:
        image_dir = split_dirs(dataset_dir, split)[0]
        if not image_dir.is_dir():
            print(f"Skipping the '{split}' split: folder not found -> {image_dir}")
            continue
        stats.append(analyse_split(dataset_dir, split, prefix_separator))
    if not any(s.images for s in stats):
        raise SystemExit(f"No .jpg / .png images found in the splits {splits} of {dataset_dir}")
    return stats


def shared_videos(stats: list[SplitStats]) -> dict[str, list[str]]:
    """Videos that are in more than one split: {video: [splits]}. Their frames leak between the splits."""
    where: dict[str, list[str]] = {}
    for s in stats:
        for video in s.videos:
            where.setdefault(video, []).append(s.name)
    return {video: splits for video, splits in where.items() if len(splits) > 1 and video != NO_VIDEO}


def size_distribution(sizes: np.ndarray) -> dict[str, int]:
    """Boxes per size range of SIZE_BINS, e.g. {"<8": 10, "8-16": 250, ..., ">=96": 0}."""
    counts = np.histogram(sizes, bins=SIZE_BINS)[0]
    names = [f"<{SIZE_BINS[1]}"] + [f"{lo}-{hi}" for lo, hi in zip(SIZE_BINS[1:-2], SIZE_BINS[2:-1])] \
        + [f">={SIZE_BINS[-2]}"]
    return dict(zip(names, (int(c) for c in counts)))


def percentiles(values: np.ndarray) -> dict[str, float] | None:
    """min, 5%, median, mean, 95% and max of the values; None if there are none."""
    if not len(values):
        return None
    p5, p50, p95 = np.percentile(values, [5, 50, 95])
    return {"min": float(values.min()), "p5": float(p5), "median": float(p50), "mean": float(values.mean()),
            "p95": float(p95), "max": float(values.max())}


def dataset_summary(stats: list[SplitStats], class_names: dict[int, str] | None = None) -> dict:
    """The statistics as plain numbers, for summary.json."""
    names = class_names or {}
    out = {"splits": {}, "videos_in_several_splits": shared_videos(stats)}
    for s in stats:
        w, h = s.boxes[:, 3], s.boxes[:, 4]
        out["splits"][s.name] = {
            "image_dir": str(s.image_dir),
            "label_dir": str(s.label_dir),
            "images": s.images,
            "images_with_objects": s.images - s.background,
            "background_images": s.background,
            "images_without_label_file": s.missing_labels,
            "label_files_without_image": s.orphan_labels,
            "unreadable_images": s.unreadable_images,
            "instances": sum(s.instances.values()),
            "instances_by_class": {str(names.get(c, c)): n for c, n in sorted(s.instances.items())},
            "instances_per_image": percentiles(s.boxes_per_image),
            "boxes_outside_image": s.out_of_bounds,
            "image_sizes": {f"{iw}x{ih}": n for (iw, ih), n in s.image_sizes.most_common()},
            "box_width_px": percentiles(w),
            "box_height_px": percentiles(h),
            "box_size_px": percentiles(s.box_sizes()),
            "box_aspect_ratio": percentiles(w / h) if len(w) else None,
            "boxes_by_size_px": size_distribution(s.box_sizes()),
            "videos": {video: {"images": v.images, "background_images": v.background,
                               "instances": sum(v.instances.values()),
                               "instances_by_class": {str(names.get(c, c)): n for c, n in sorted(v.instances.items())}}
                       for video, v in s.videos.items()},
        }
    return out
