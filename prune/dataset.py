"""The files of the dataset being pruned: its label files, the video each belongs to and the
image that goes with it.

The dataset is a YOLO dataset laid out as images/<split>/ and labels/<split>/ (the pruned copy
is built from those two folders). A frame's video prefix is found as in the dataset analysis
(dataset_tools.image_tools.analysis.dataset_stats.video_prefix).
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from common.yolo_format import IMAGE_SUFFIXES
from dataset_tools.image_tools.analysis.dataset_stats import video_prefix

ALL_PREFIXES = "ALL"


def check_layout(root_dir):
    """Stop with a message unless the dataset is laid out as images/<split>/ + labels/<split>/."""
    for sub in ("images", "labels"):
        if not os.path.isdir(os.path.join(root_dir, sub)):
            raise SystemExit(f"Dataset pruning needs a YOLO dataset laid out as images/<split>/ and "
                             f"labels/<split>/: no '{sub}' folder in {root_dir}")


def label_prefix(label_path, lbl_root):
    """The video prefix of a label file. Every module must find it this way, so their groups match."""
    return video_prefix(Path(label_path), Path(lbl_root))


def find_image(label_path, lbl_root, img_root):
    """Mirror the label's relative path under images/<split>/ and try each extension."""
    rel_stem = os.path.splitext(os.path.relpath(label_path, lbl_root))[0]
    for ext in IMAGE_SUFFIXES:
        img_path = os.path.join(img_root, rel_stem + ext)
        if os.path.exists(img_path):
            return img_path
    return None


def get_label_files(root_dir, split, prefix=ALL_PREFIXES):
    """All label files under labels/<split>/ belonging to the prefix (every prefix if ALL).
    The fingerprint is computed from this list, so all modules must build it the same way."""
    lbl_root = os.path.join(root_dir, "labels", split)
    label_files = []
    for dirpath, _, filenames in os.walk(lbl_root):
        for lbl_name in filenames:
            if not lbl_name.endswith('.txt'):
                continue
            if prefix == ALL_PREFIXES or label_prefix(os.path.join(dirpath, lbl_name), lbl_root) == prefix:
                label_files.append(os.path.join(dirpath, lbl_name))
    return label_files


def group_by_prefix(label_files, lbl_root):
    """{video prefix: its label files}, by the same prefix rule as get_label_files(), so the groups
    match the analysed files."""
    groups = defaultdict(list)
    for label_path in label_files:
        groups[label_prefix(label_path, lbl_root)].append(label_path)
    return groups


def read_label_boxes(label_path):
    """Returns the label's boxes as (x_c, y_c, w, h) floats, or None if any line is unparseable."""
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 5:
                return None
            try:
                boxes.append(tuple(float(p) for p in parts[1:5]))
            except ValueError:
                return None
    return boxes


def count_dataset(root_dir):
    """Images, frames (label files) and instances (box lines) per split, plus frames and
    instances per video prefix, counted from disk."""
    counts = {}
    img_dir = os.path.join(root_dir, "images")
    for split in sorted(d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))):
        n_images = sum(1 for _, _, filenames in os.walk(os.path.join(img_dir, split))
                       for name in filenames if name.endswith(IMAGE_SUFFIXES))
        lbl_root = os.path.join(root_dir, "labels", split)
        prefixes = {}
        for prefix, files in group_by_prefix(get_label_files(root_dir, split), lbl_root).items():
            instances = 0
            for label_path in tqdm(files, desc=f"Counting {os.path.basename(root_dir)} {split}", unit="file", leave=False):
                with open(label_path, 'r') as f:
                    instances += sum(1 for line in f if len(line.split()) >= 5)
            prefixes[prefix] = {"frames": len(files), "instances": instances}
        counts[split] = {
            "images": n_images,
            "frames": sum(p["frames"] for p in prefixes.values()),
            "instances": sum(p["instances"] for p in prefixes.values()),
            "prefixes": prefixes,
        }
    return counts
