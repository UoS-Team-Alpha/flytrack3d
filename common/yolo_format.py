"""Reading a YOLO detection dataset from disk: its split folders, images and label files.

Label files are read either as they are written (read_yolo_boxes: normalised centre boxes)
or as pixel corner boxes for an image of a known size (read_yolo_labels).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = (".jpg", ".png")


def split_dirs(dataset_dir: str | Path, split: str) -> tuple[Path, Path]:
    """(image folder, label folder) of a split: images/<split> + labels/<split>, or <split>/images + <split>/labels."""
    dataset_dir = Path(dataset_dir)
    images, labels = dataset_dir / "images" / split, dataset_dir / "labels" / split
    if not images.exists():
        images = dataset_dir / split / "images"
    if not labels.exists():
        labels = dataset_dir / split / "labels"
    return images, labels


def list_images(image_dir: str | Path) -> list[Path]:
    """The .jpg and .png images of a folder, sorted by file name."""
    return sorted(p for p in Path(image_dir).glob("*") if p.suffix in IMAGE_SUFFIXES)


def read_yolo_boxes(label_path: str | Path) -> list[list[float]]:
    """The [class, x_centre, y_centre, width, height] rows of a label file (normalised); [] if it is missing."""
    label_path = Path(label_path)
    if not label_path.exists():
        return []
    rows = (line.split() for line in label_path.read_text().splitlines())
    return [[float(v) for v in row] for row in rows if len(row) == 5]


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
