"""Reading a YOLO detection dataset from disk: its split folders, images and label files.

Label files are read either as they are written (read_yolo_boxes: normalised centre boxes)
or as pixel corner boxes for an image of a known size (read_yolo_labels).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = (".jpg", ".png")
SPLIT_NAMES = ("train", "val", "valid", "validation", "test")


def split_dirs(dataset_dir: str | Path, split: str) -> tuple[Path, Path]:
    """(image folder, label folder) of a split: images/<split> + labels/<split>, or <split>/images + <split>/labels."""
    dataset_dir = Path(dataset_dir)
    images, labels = dataset_dir / "images" / split, dataset_dir / "labels" / split
    if not images.exists():
        images = dataset_dir / split / "images"
    if not labels.exists():
        labels = dataset_dir / split / "labels"
    return images, labels


def find_splits(dataset_dir: str | Path) -> list[str]:
    """The splits (of SPLIT_NAMES) a dataset has an image folder for."""
    return [split for split in SPLIT_NAMES if split_dirs(dataset_dir, split)[0].is_dir()]


def list_images(image_dir: str | Path, recursive: bool = False) -> list[Path]:
    """The .jpg and .png images of a folder, sorted by file name. recursive = also those in its sub-folders."""
    return sorted(p for p in Path(image_dir).glob("**/*" if recursive else "*") if p.suffix in IMAGE_SUFFIXES)


def read_yolo_rows(label_path: str | Path) -> np.ndarray:
    """(N, 6) class, x_centre, y_centre, width, height, conf (normalised) of a label file: ground
    truth ("class xc yc w h", conf = 1) or predictions ("class xc yc w h conf"). Rows of any other
    length (e.g. segmentation polygons) are skipped. Empty if the file is missing."""
    label_path = Path(label_path)
    if not label_path.is_file():
        return np.zeros((0, 6))
    rows = [ln.split() for ln in label_path.read_text().splitlines()]
    rows = [r + ["1"] if len(r) == 5 else r for r in rows if len(r) in (5, 6)]
    return np.array(rows, dtype=np.float64).reshape(-1, 6)


def read_yolo_boxes(label_path: str | Path) -> list[list[float]]:
    """The [class, x_centre, y_centre, width, height] rows of a label file (normalised; a prediction
    file's conf column is dropped); [] if it is missing."""
    return read_yolo_rows(label_path)[:, :5].tolist()


def read_yolo_labels(path: Path, width: int, height: int) -> np.ndarray:
    """Read a YOLO txt file (class xc yc w h [conf]) into (N, 5) class + pixel xyxy (conf dropped)."""
    cls, xc, yc, w, h, _ = read_yolo_rows(path).T
    return np.stack([
        cls,
        (xc - w / 2) * width,
        (yc - h / 2) * height,
        (xc + w / 2) * width,
        (yc + h / 2) * height,
    ], axis=1)
