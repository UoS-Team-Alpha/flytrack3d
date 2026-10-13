"""Look at what the redundancy analysis found and what a prune run drops. The frames are chosen
here and drawn by visualise.boxes.pruning.

    view_rank_grids        pick saved bbox results, then a rank: sample frames holding that box size
    review_prune_manifest  pick a manifest: dropped frames next to the kept frame each one matched

Both are interactive; q quits at any prompt and Ctrl+C exits cleanly, also while a window is open.
"""

from __future__ import annotations

import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from common.yolo_format import read_yolo_boxes
from visualise.boxes.pruning import show_box_samples, show_dropped_pairs

from .dataset import find_image, get_label_files
from .paths import dataset_name, get_bbox_output_dir, get_purge_output_dir
from .prompts import prompt_choice
from .redundancy import bbox_size, parse_decimals
from .runs import load_manifest


def read_label_bboxes(label_files):
    """[(label_path, [[cls, x_c, y_c, w, h], ...]), ...] in frame order, skipping files without boxes."""
    label_bboxes = []
    for label_path in tqdm(sorted(label_files), desc="Reading labels", unit="file"):
        bboxes = read_yolo_boxes(label_path)
        if bboxes:
            label_bboxes.append((label_path, bboxes))
    return label_bboxes


def show_rank_grid(label_bboxes, row, root_dir, n=9, grid=True, close_text="choose another rank"):
    """Show n images sampled uniformly from those containing the rank's bbox size.
    row is a rank row of a bbox results CSV. Returns False if no images were found."""
    split, prefix = row["split"], row["video_prefix"]
    decimals = parse_decimals(row.get("decimals"))
    img_root = os.path.join(root_dir, "images", split)
    lbl_root = os.path.join(root_dir, "labels", split)
    target = (float(row["width"]), float(row["height"]))

    # Find all images containing a bbox of the selected size
    matches = []
    for label_path, bboxes in label_bboxes:
        matching = [box for box in bboxes if bbox_size(box[3], box[4], decimals) == target]
        if matching:
            img_path = find_image(label_path, lbl_root, img_root)
            if img_path:
                matches.append((img_path, matching))

    if not matches:
        print(f"\nNo images found with bbox size {target}.")
        return False

    # Sample n images uniformly across the (sorted) matches
    indices = np.unique(np.linspace(0, len(matches) - 1, min(n, len(matches))).round().astype(int))
    samples = [matches[i] for i in indices]
    names = [os.path.relpath(img_path, img_root) for img_path, _ in samples]

    print(f"\nRank {row['rank']}: found {len(matches)} images with bbox size {target[0]} x {target[1]}. "
          f"Showing {len(samples)} sampled uniformly. Close the window to {close_text}.")

    distinct = f" | distinct {row['distinct']}/{row['count']}" if row.get("distinct") else ""
    title = f"Rank {row['rank']} | {target[0]} x {target[1]}{distinct} | {split} | {prefix}"
    show_box_samples(samples, names, title, grid, window_title=f"Rank {row['rank']} | {prefix}")
    return True


def view_rank_grids(root_dir, n=9, grid=True):
    """Pick saved bbox results of this dataset, then a rank, to see n of its frames; repeat until q."""
    try:
        _view_rank_grids(root_dir, n, grid)
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
    finally:
        plt.close('all')


def _view_rank_grids(root_dir, n, grid):
    dataset = dataset_name(root_dir)

    # Step 1: Select a bbox size CSV saved by the redundancy analysis for this dataset
    bbox_dir = get_bbox_output_dir()
    csv_options = []
    for csv_name in sorted(os.listdir(bbox_dir)):
        if not csv_name.endswith('.csv') or csv_name.endswith('_decisions.csv'):
            continue
        with open(os.path.join(bbox_dir, csv_name), 'r', newline='') as f:
            rows = list(csv.DictReader(f))
        if rows and rows[0].get("dataset") == dataset:
            csv_options.append((csv_name, rows))

    if not csv_options:
        print(f"No bbox size CSVs found for dataset '{dataset}' in -> {bbox_dir}")
        print("Analyse a video prefix with saving on first.")
        return

    print("\n" + "=" * 65)
    print(f" BBOX SIZE RESULTS | DATASET: {dataset}")
    print("=" * 65)
    print(f"{'#':<4} | {'Split':<6} | {'Video Prefix'}")
    print("-" * 65)
    for i, (csv_name, rows) in enumerate(csv_options, 1):
        print(f"{i:<4} | {rows[0]['split']:<6} | {rows[0]['video_prefix']}")

    choice = prompt_choice(csv_options, "\nSelect results number (q to quit): ")
    if choice is None:
        return
    csv_name, rows = choice
    split = rows[0]["split"]
    prefix = rows[0]["video_prefix"]

    # Step 2: Read all bboxes for this split/prefix once, so switching rank is fast
    label_bboxes = read_label_bboxes(get_label_files(root_dir, split, prefix))

    # Step 3: Repeatedly select a rank and view it, until the user quits
    while True:
        print("\n" + "=" * 65)
        print(f" SPLIT: {split.upper()} | PREFIX: {prefix}")
        print("=" * 65)
        print(f"{'Rank':<5} | {'Width':<10} | {'Height':<10} | {'Count':<7} | {'% of BBoxes':<11} | "
              f"{'Frames':<7} | {'Distinct'}")
        print("-" * 65)
        for row in rows:
            print(f"{row['rank']:<5} | {row['width']:<10} | {row['height']:<10} | {row['count']:<7} | "
                  f"{row['pct_of_bboxes']:<11} | {row['frames']:<7} | {row.get('distinct') or '-'}")

        row = prompt_choice(rows, "\nSelect rank (q to quit): ")
        if row is None:
            return
        show_rank_grid(label_bboxes, row, root_dir, n, grid)


def review_prune_manifest(root_dir, n=6):
    """Show frames a prune run drops next to the kept frame each one matched: first n evenly
    spaced ones, then random samples until q."""
    try:
        _review_prune_manifest(root_dir, n)
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
    finally:
        plt.close('all')


def _review_prune_manifest(root_dir, n):
    dataset = dataset_name(root_dir)

    # Step 1: Select a prune manifest for this dataset
    purge_dir = get_purge_output_dir()
    manifests = []
    for name in sorted(os.listdir(purge_dir), reverse=True):
        if not (name.startswith("prune_") and name.endswith(".csv")):
            continue
        rows = load_manifest(name)
        if rows and rows[0]["dataset"] == dataset:
            manifests.append((name, rows))

    if not manifests:
        print(f"No prune manifests found for dataset '{dataset}' in -> {purge_dir}")
        print("Accept a video prefix in guided pruning first.")
        return

    print("\n" + "=" * 65)
    print(f" PRUNE MANIFESTS | DATASET: {dataset}")
    print("=" * 65)
    print(f"{'#':<4} | {'Split':<6} | {'Frames':<7} | {'Run ID'}")
    print("-" * 65)
    for i, (name, rows) in enumerate(manifests, 1):
        print(f"{i:<4} | {rows[0]['split']:<6} | {len({r['label'] for r in rows}):<7} | {rows[0]['run_id']}")

    choice = prompt_choice(manifests, "\nSelect manifest number (q to quit): ")
    if choice is None:
        return
    name, rows = choice
    split = rows[0]["split"]
    img_root = os.path.join(root_dir, "images", split)
    lbl_root = os.path.join(root_dir, "labels", split)

    # Group box rows by frame (manifest order is frame order)
    frames = {}
    for row in rows:
        frames.setdefault(row["label"], []).append(row)
    labels = list(frames)

    # Step 2: First sample evenly spaced, then random samples until the user quits
    rng = np.random.default_rng()
    indices = np.unique(np.linspace(0, len(labels) - 1, min(n, len(labels))).round().astype(int))
    while True:
        pairs = []
        for i in indices:
            frame_rows = frames[labels[i]]
            first = frame_rows[0]
            matched_path = os.path.join(lbl_root, first["matched_label"])
            matched_img = find_image(matched_path, lbl_root, img_root)
            matched_box = (float(first["matched_x_center"] or 0.5), float(first["matched_y_center"] or 0.5),
                           float(first["width"]), float(first["height"]))
            pairs.append({
                "img_path": os.path.join(img_root, first["image"]),
                "bboxes": [tuple(float(r[k]) for k in ("x_center", "y_center", "width", "height"))
                           for r in frame_rows],
                "box": tuple(float(first[k]) for k in ("x_center", "y_center", "width", "height")),
                "rank": first["rank"],
                "ssim": first["matched_ssim"],
                "name": os.path.splitext(first["label"])[0],
                "matched_img_path": matched_img,
                "matched_bboxes": (read_yolo_boxes(matched_path) or [matched_box]) if matched_img else [],
                "matched_box": matched_box,
                "matched_name": os.path.splitext(first["matched_label"])[0],
            })

        print(f"\n{len(labels)} dropped frames in {name}. Showing {len(pairs)}. "
              f"Close the window to continue.")
        show_dropped_pairs(pairs, f"{split} | {rows[0]['run_id']}")

        if input("\nEnter for a random sample, q to quit: ").strip().lower() == 'q':
            return
        indices = np.sort(rng.choice(len(labels), size=min(n, len(labels)), replace=False))
