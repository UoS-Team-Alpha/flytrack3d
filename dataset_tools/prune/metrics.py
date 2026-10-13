"""Frames and instances per split and video prefix, saved as a CSV that the prompts and the
purge status table read. The numbers are the dataset analysis's
(dataset_tools.stats); the CSV saves counting them again on every
run, so refresh it when images are added to or removed from the dataset.
"""

from __future__ import annotations

import csv
import os

from common.datasets import read_class_names
from dataset_tools.stats import analyse_dataset

from .dataset import ALL_PREFIXES
from .paths import get_metrics_csv_path
from .prompts import prompt_choice

FIXED_COLS = ("split", "video_prefix", "frames", "total_instances", "missing_labels")


def write_video_metrics(root_dir, splits=None):
    """Count the dataset (splits = None: every split it has) and save the metrics CSV: one row
    per video prefix of each split, most frames first, then the split's TOTAL row."""
    stats = analyse_dataset(root_dir, splits=splits)
    data_yaml = os.path.join(root_dir, "data.yaml")
    class_names = read_class_names(data_yaml if os.path.isfile(data_yaml) else None)
    class_ids = sorted({c for s in stats for c in s.instances})
    class_cols = [str(class_names.get(c, f"Class {c}")) for c in class_ids]

    rows = []
    for s in stats:
        for video, v in sorted(s.videos.items(), key=lambda kv: -kv[1].images):
            rows.append({"split": s.name, "video_prefix": video, "frames": v.images,
                         "total_instances": sum(v.instances.values()), "missing_labels": "",
                         **{col: v.instances.get(c, 0) for c, col in zip(class_ids, class_cols)}})
        rows.append({"split": s.name, "video_prefix": "TOTAL", "frames": s.images,
                     "total_instances": sum(s.instances.values()), "missing_labels": s.missing_labels,
                     **{col: s.instances.get(c, 0) for c, col in zip(class_ids, class_cols)}})

    csv_path = get_metrics_csv_path(root_dir)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(FIXED_COLS) + class_cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nMetrics saved to: {csv_path}")
    return csv_path


def load_metrics_rows(root_dir):
    """Rows of the metrics CSV written by write_video_metrics(), or None if there is none yet."""
    csv_path = get_metrics_csv_path(root_dir)
    if not os.path.exists(csv_path):
        print(f"Metrics CSV not found -> {csv_path}")
        return None
    with open(csv_path, 'r', newline='') as f:
        return list(csv.DictReader(f))


def prompt_split(rows, splits=None):
    """Select a split by number (split-wide totals shown to guide the choice).
    splits limits the options; returns None if the user quits."""
    split_options = [s for s in dict.fromkeys(row["split"] for row in rows) if splits is None or s in splits]
    split_totals = {row["split"]: row for row in rows if row["video_prefix"] == "TOTAL"}
    if not split_options:
        print(f"None of the splits {list(splits)} are in the metrics CSV.")
        return None

    print("\n" + "=" * 65)
    print(" AVAILABLE SPLITS")
    print("=" * 65)
    print(f"{'#':<4} | {'Split':<10} | {'Frames':<8} | {'Total Inst':<10} | {'Missing'}")
    print("-" * 65)
    for i, split in enumerate(split_options, 1):
        total = split_totals.get(split, {})
        print(f"{i:<4} | {split:<10} | {total.get('frames', '-'):<8} | "
              f"{total.get('total_instances', '-'):<10} | {total.get('missing_labels', '-')}")

    return prompt_choice(split_options, "\nSelect split number (q to quit): ")


def prompt_prefix(rows, split):
    """Display the metrics for the split, then select a video prefix (or all of them)"""
    split_rows = [row for row in rows if row["split"] == split and row["video_prefix"] != "TOTAL"]
    split_totals = {row["split"]: row for row in rows if row["video_prefix"] == "TOTAL"}
    class_cols = [c for c in rows[0].keys() if c not in FIXED_COLS]

    print("\n" + "=" * 65)
    print(f" SPLIT: {split.upper()}")
    print("=" * 65)
    print(f"{'#':<4} | {'Video Prefix':<25} | {'Frames':<8} | {'Total Inst':<10} | {'Class Breakdown'}")
    print("-" * 65)
    print(f"{1:<4} | {'<all prefixes>':<25} | {split_totals[split]['frames']:<8} | "
          f"{split_totals[split]['total_instances']:<10} |")
    for i, row in enumerate(split_rows, 2):
        breakdown_str = ", ".join(f"{col}: {row[col]}" for col in class_cols)
        print(f"{i:<4} | {row['video_prefix']:<25} | {row['frames']:<8} | "
              f"{row['total_instances']:<10} | {breakdown_str}")

    prefixes = [ALL_PREFIXES] + [row["video_prefix"] for row in split_rows]
    return prompt_choice(prefixes, "\nSelect video prefix number (q to quit): ")
