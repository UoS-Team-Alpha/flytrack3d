"""Where the pruning step keeps its files. Everything is under one folder, by default
output/pruned_datasets/ of the repository (the main_*.py scripts point it at the config's
output folder with set_output_dir):

    <dataset>_pruned_<ddmmyyyy>/         the pruned datasets, and the summary of each guided build
    <dataset>_metrics.csv                frames and instances per split and video prefix
    bbox_sizes/                          bbox size ranks, SSIM decisions and their fingerprints
    purge/                               manifests, the purge log and the guided sessions

The folder must be on the same filesystem as the dataset for the images to be hard-linked.
"""

from __future__ import annotations

import os
from pathlib import Path

_output_dir = Path(__file__).resolve().parents[1] / "output" / "pruned_datasets"


def set_output_dir(path: str | Path) -> None:
    global _output_dir
    _output_dir = Path(path).resolve()


def get_output_dir() -> str:
    os.makedirs(_output_dir, exist_ok=True)
    return str(_output_dir)


def get_metrics_csv_path(root_dir) -> str:
    return os.path.join(get_output_dir(), f"{dataset_name(root_dir)}_metrics.csv")


def get_bbox_output_dir() -> str:
    """BBox size CSVs are kept in their own subfolder, apart from the metrics CSV."""
    bbox_dir = os.path.join(get_output_dir(), "bbox_sizes")
    os.makedirs(bbox_dir, exist_ok=True)
    return bbox_dir


def get_purge_output_dir() -> str:
    purge_dir = os.path.join(get_output_dir(), "purge")
    os.makedirs(purge_dir, exist_ok=True)
    return purge_dir


def get_purge_log_path() -> str:
    return os.path.join(get_purge_output_dir(), "purge_log.csv")


def dataset_name(root_dir) -> str:
    """A dataset is known by its folder name in every file the pruning step writes."""
    return os.path.basename(os.path.normpath(root_dir))
