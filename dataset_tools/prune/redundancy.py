"""Find the redundant boxes of a video: boxes of the same size that also look the same.

A fly that sits still gets the same box size in hundreds of consecutive frames (augmentation
moves a box but does not resize it), so the boxes of a split / video prefix are grouped by
their exact width and height. Each size is a rank: rank 1 is the most common size. For the top
ranks, each box is cropped from the greyscale image and compared with SSIM in frame order: a
crop is kept if it differs (SSIM below the threshold) from every crop kept so far, else it is
redundant. A rank's distinct / count ratio says how varied it is.

Results are saved to bbox_sizes/ (see paths) as the rank table, the per-box decisions and a
fingerprint of the label files and parameters, by which saved results are reused or refused as
out of date. Nothing here changes the dataset.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime

import numpy as np
from PIL import Image
from tqdm import tqdm

from .dataset import find_image, get_label_files
from .metrics import load_metrics_rows, prompt_prefix, prompt_split
from .paths import dataset_name, get_bbox_output_dir

SSIM_WIN_SIZE = 7  # skimage default window; crops are padded to at least this size


def bbox_size(width, height, decimals=None):
    """Decimals=None compares exact values, so only identical sizes are grouped."""
    if decimals is None:
        return (width, height)
    return (round(width, decimals), round(height, decimals))


def parse_decimals(value):
    """Read the 'decimals' column back from a bbox CSV (missing in older CSVs, which used 6)"""
    if value is None:
        return 6
    if value in ("", "none"):
        return None
    return int(value)


def crop_bbox(gray, x_c, y_c, w, h):
    """Crop a YOLO bbox from a greyscale image, padded to at least the SSIM window."""
    img_h, img_w = gray.shape
    crop_w = min(max(SSIM_WIN_SIZE, round(w * img_w)), img_w)
    crop_h = min(max(SSIM_WIN_SIZE, round(h * img_h)), img_h)
    x0 = min(max(0, round(x_c * img_w - crop_w / 2)), img_w - crop_w)
    y0 = min(max(0, round(y_c * img_h - crop_h / 2)), img_h - crop_h)
    return gray[y0:y0 + crop_h, x0:x0 + crop_w]


def compute_ssim_stats(rank_boxes, lbl_root, img_root, threshold):
    """Rank_boxes: {rank: [(label_path, x_c, y_c, w, h), ...]}
    Returns (stats, decisions):
      stats:     {rank: {"distinct": int, "mean_ssim": float, "min_ssim": float}}
      decisions: one dict per box - kept (a new appearance) or redundant (matches
                 an earlier kept box), so pruning uses exactly the Distinct logic."""
    from skimage.metrics import structural_similarity

    # Group boxes by image so each image is read only once for all ranks
    image_boxes = {}
    for rank, boxes in rank_boxes.items():
        for label_path, x_c, y_c, w, h in boxes:
            image_boxes.setdefault(label_path, []).append((rank, x_c, y_c, w, h))

    crops = {rank: [] for rank in rank_boxes}
    decisions = []
    for label_path in tqdm(sorted(image_boxes), desc="Cropping bboxes for SSIM", unit="img"):
        img_path = find_image(label_path, lbl_root, img_root)
        if img_path is None:
            for rank, x_c, y_c, w, h in image_boxes[label_path]:
                decisions.append(make_decision(rank, label_path, None, (x_c, y_c, w, h),
                                               "missing_image", lbl_root, img_root))
            continue
        gray = np.asarray(Image.open(img_path).convert('L'))
        for rank, x_c, y_c, w, h in image_boxes[label_path]:
            crop = crop_bbox(gray, x_c, y_c, w, h)
            # Keep crops in a group the same shape (only differs if resolutions are mixed)
            if crops[rank] and crop.shape != crops[rank][0]["crop"].shape:
                ref_h, ref_w = crops[rank][0]["crop"].shape
                crop = np.asarray(Image.fromarray(crop).resize((ref_w, ref_h)))
            crops[rank].append({"label_path": label_path, "img_path": img_path,
                                "box": (x_c, y_c, w, h), "crop": crop})

    def ssim(a, b):
        return structural_similarity(a, b, data_range=255, win_size=SSIM_WIN_SIZE)

    stats = {}
    for rank, rank_crops in crops.items():
        if not rank_crops:
            continue

        consecutive = []
        kept = []
        pbar = tqdm(rank_crops, desc=f"SSIM rank {rank}/{len(crops)}", unit="crop")
        for i, item in enumerate(pbar):
            crop = item["crop"]
            # Similarity between consecutive frames (crops are in frame order)
            if i > 0:
                consecutive.append(ssim(rank_crops[i - 1]["crop"], crop))

            # Distinct: a crop counts as new if it differs from every crop kept so far.
            # Check the most recently kept first, as neighbouring frames match most often.
            match, match_score = None, None
            for k in reversed(kept):
                score = ssim(crop, k["crop"])
                if score >= threshold:
                    match, match_score = k, score
                    break

            if match is None:
                kept.append(item)
                pbar.set_postfix(distinct=len(kept))
                decisions.append(make_decision(rank, item["label_path"], item["img_path"], item["box"],
                                               "kept", lbl_root, img_root))
            else:
                decisions.append(make_decision(rank, item["label_path"], item["img_path"], item["box"],
                                               "redundant", lbl_root, img_root,
                                               matched=match, score=match_score))

        stats[rank] = {
            "distinct": len(kept),
            "mean_ssim": round(float(np.mean(consecutive)), 3) if consecutive else None,
            "min_ssim": round(float(np.min(consecutive)), 3) if consecutive else None,
        }

    decisions.sort(key=lambda d: (d["rank"], d["label"]))
    return stats, decisions


def make_decision(rank, label_path, img_path, box, decision, lbl_root, img_root, matched=None, score=None):
    """Paths are stored relative to labels/<split>/ and images/<split>/ so the file stays portable."""
    x_c, y_c, w, h = box
    return {
        "rank": rank,
        "label": os.path.relpath(label_path, lbl_root),
        "image": os.path.relpath(img_path, img_root) if img_path else "",
        "x_center": x_c,
        "y_center": y_c,
        "width": w,
        "height": h,
        "decision": decision,
        "matched_label": os.path.relpath(matched["label_path"], lbl_root) if matched else "",
        "matched_ssim": round(float(score), 4) if score is not None else "",
    }


def get_decisions_csv_path(bbox_csv_path):
    """Per-box SSIM decisions sit next to the rank CSV they belong to."""
    return os.path.splitext(bbox_csv_path)[0] + "_decisions.csv"


def get_bbox_csv_path(root_dir, split, prefix, prefix_len=21):
    """Named distinctly from the metrics CSV so it is never read back as input.
    The prefix is truncated for readability; a short hash of the full
    dataset/split/prefix keeps the name unique."""
    short_prefix = prefix[:prefix_len].rstrip("_-")
    name_hash = hashlib.sha1(f"{dataset_name(root_dir)}/{split}/{prefix}".encode()).hexdigest()[:6]
    return os.path.join(get_bbox_output_dir(), f"bbox_{split}_{short_prefix}_{name_hash}.csv")


def get_results_fingerprint(label_files, params):
    """Fingerprint of the label files (path, size, modified time) plus the analysis
    parameters. If either changes, the saved results are out of date."""
    digest = hashlib.sha1(json.dumps(params, sort_keys=True).encode())
    for label_path in sorted(label_files):
        stat = os.stat(label_path)
        digest.update(f"{label_path}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def load_cached_results(csv_path, fingerprint):
    """Returns (rows, meta) if saved results exist for this exact fingerprint, else None."""
    meta_path = os.path.splitext(csv_path)[0] + ".json"
    if not (os.path.exists(csv_path) and os.path.exists(meta_path)):
        return None
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    if meta.get("fingerprint") != fingerprint:
        return None
    # SSIM results are only complete if their decisions file is still there
    if meta["params"].get("ssim") and not os.path.exists(get_decisions_csv_path(csv_path)):
        return None
    with open(csv_path, 'r', newline='') as f:
        return list(csv.DictReader(f)), meta


def print_bbox_table(rows, summary, params, split, prefix):
    ssim = params["ssim"]
    line_width = 100 if ssim else 65

    def fmt(value):
        return "-" if value in (None, "") else value

    print("\n" + "=" * line_width)
    print(f" TOP {params['top_n']} BBOX SIZES | SPLIT: {split.upper()} | PREFIX: {prefix}")
    print("=" * line_width)
    header = f"{'Rank':<5} | {'Width':<10} | {'Height':<10} | {'Count':<7} | {'% of BBoxes':<11} | {'Frames':<7}"
    if ssim:
        header += f" | {'Distinct':<8} | {'Mean SSIM':<9} | {'Min SSIM'}"
    print(header)
    print("-" * line_width)
    for row in rows:
        line = (f"{row['rank']:<5} | {row['width']:<10} | {row['height']:<10} | {row['count']:<7} | "
                f"{row['pct_of_bboxes']:<11} | {row['frames']:<7}")
        if ssim:
            line += f" | {fmt(row['distinct']):<8} | {fmt(row['mean_ssim']):<9} | {fmt(row['min_ssim'])}"
        print(line)

    print("-" * line_width)
    if ssim:
        print(f"Distinct:  bboxes whose crop differs from every earlier kept crop (SSIM < {params['ssim_threshold']})")
        print(f"Mean/Min SSIM: similarity between consecutive frames (1.0 = identical)")
        print("-" * line_width)
    total_bboxes = summary["total_bboxes"]
    duplicate_bboxes = summary["duplicate_bboxes"]
    print(f"Label files scanned:        {summary['label_files']}")
    print(f"Total bboxes:               {total_bboxes}")
    print(f"Unique bbox sizes:          {summary['unique_sizes']}")
    print(f"BBoxes sharing a size:      {duplicate_bboxes} ({100 * duplicate_bboxes / total_bboxes:.2f}%)")


def analyse_bbox_duplicates(root_dir, top_n=10, decimals=None, save_csv=False,
                            ssim=False, ssim_threshold=0.9, ssim_ranks=None, use_cache=True):
    """Analyse a split and video prefix chosen in the terminal (see run_bbox_analysis).
    ssim=True adds SSIM stats for the top ssim_ranks ranks (all top_n if None).
    use_cache=True shows saved results instead of recomputing, if the label files
    and parameters are unchanged since they were saved."""
    rows = load_metrics_rows(root_dir)
    if rows is None:
        return

    # Step 1: Select the split, then the video prefix
    split = prompt_split(rows)
    if split is None:
        return
    prefix = prompt_prefix(rows, split)
    if prefix is None:
        return

    # Step 2: Analyse (or reuse saved results)
    run_bbox_analysis(root_dir, split, prefix, top_n=top_n, decimals=decimals, save_csv=save_csv,
                      ssim=ssim, ssim_threshold=ssim_threshold, ssim_ranks=ssim_ranks, use_cache=use_cache)


def run_bbox_analysis(root_dir, split, prefix, top_n=10, decimals=None, save_csv=False, ssim=False,
                      ssim_threshold=0.9, ssim_ranks=None, use_cache=True, label_files=None):
    """Non-interactive bbox size analysis for one split/prefix. label_files can be passed in
    to avoid walking the split again (it must be the same list get_label_files() returns).
    Returns {"csv_path", "fingerprint", "params", "summary"}, or None if there are no bboxes."""
    lbl_root = os.path.join(root_dir, "labels", split)
    if label_files is None:
        label_files = get_label_files(root_dir, split, prefix)

    # Reuse saved results if the label files and parameters are unchanged
    params = {
        "top_n": top_n,
        "decimals": decimals,
        "ssim": ssim,
        "ssim_threshold": ssim_threshold if ssim else None,
        "ssim_ranks": ssim_ranks if ssim else None,
    }
    out_path = get_bbox_csv_path(root_dir, split, prefix)
    fingerprint = get_results_fingerprint(label_files, params)
    result = {"csv_path": out_path, "fingerprint": fingerprint, "params": params}

    if use_cache:
        cached = load_cached_results(out_path, fingerprint)
        if cached:
            rows, meta = cached
            print(f"\nUsing saved results from {meta['created']} (labels and parameters unchanged).")
            print_bbox_table(rows, meta["summary"], params, split, prefix)
            print(f"\nResults file: {out_path}")
            return {**result, "summary": meta["summary"]}

    # Step 1: Tally bbox sizes (YOLO format: class x_center y_center width height)
    size_counts = Counter()
    size_frames = {}
    size_boxes = {}
    total_bboxes = 0

    for label_path in tqdm(label_files, desc=f"Tallying bboxes ({prefix})", unit="file"):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    x_c, y_c, w, h = (float(p) for p in parts[1:5])
                except ValueError:
                    continue

                size = bbox_size(w, h, decimals)
                size_counts[size] += 1
                size_frames.setdefault(size, set()).add(os.path.basename(label_path))
                if ssim:
                    size_boxes.setdefault(size, []).append((label_path, x_c, y_c, w, h))
                total_bboxes += 1

    if total_bboxes == 0:
        print(f"\nNo bounding boxes found for split '{split}', prefix '{prefix}'.")
        return None

    top_sizes = size_counts.most_common(top_n)

    # Step 2 (optional): SSIM within the bbox region for the top ranks
    ssim_stats = {}
    if ssim:
        n_ssim = len(top_sizes) if ssim_ranks is None else min(ssim_ranks, len(top_sizes))
        rank_boxes = {rank: size_boxes[size]
                      for rank, (size, _) in enumerate(top_sizes[:n_ssim], 1)}
        img_root = os.path.join(root_dir, "images", split)
        ssim_stats, decisions = compute_ssim_stats(rank_boxes, lbl_root, img_root, ssim_threshold)

    # Step 3: Build the results rows, then print them
    rows = []
    for rank, (size, count) in enumerate(top_sizes, 1):
        width, height = size
        stats = ssim_stats.get(rank, {})
        rows.append({
            "dataset": dataset_name(root_dir),
            "split": split,
            "video_prefix": prefix,
            "rank": rank,
            "width": width,
            "height": height,
            "count": count,
            "pct_of_bboxes": round(100 * count / total_bboxes, 2),
            "frames": len(size_frames[size]),
            "decimals": "none" if decimals is None else decimals,
            "distinct": stats.get("distinct", ""),
            "mean_ssim": stats.get("mean_ssim", ""),
            "min_ssim": stats.get("min_ssim", ""),
            "ssim_threshold": ssim_threshold if stats else "",
        })

    summary = {
        "label_files": len(label_files),
        "total_bboxes": total_bboxes,
        "unique_sizes": len(size_counts),
        "duplicate_bboxes": sum(cnt for cnt in size_counts.values() if cnt > 1),
    }
    print_bbox_table(rows, summary, params, split, prefix)

    if save_csv:
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        # Per-box SSIM decisions (kept/redundant), reused by pruning instead of recomputing SSIM
        decisions_path = None
        if ssim:
            decisions_path = get_decisions_csv_path(out_path)
            with open(decisions_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(decisions[0].keys()) if decisions else ["rank"])
                writer.writeheader()
                writer.writerows(decisions)

        # Sidecar metadata used to decide whether these results can be reused
        meta_path = os.path.splitext(out_path)[0] + ".json"
        with open(meta_path, 'w') as f:
            json.dump({
                "fingerprint": fingerprint,
                "created": datetime.now().isoformat(timespec="seconds"),
                "params": params,
                "summary": summary,
                "decisions_file": os.path.basename(decisions_path) if decisions_path else None,
            }, f, indent=2)

        print(f"\nBBox size results saved to: {out_path}")
        if decisions_path:
            print(f"SSIM decisions saved to:    {decisions_path}")

    return {**result, "summary": summary}
