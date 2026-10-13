"""Work out which frames a run drops, from the SSIM decisions of redundancy.py.

Only ranks whose distinct / count ratio is below the cut-off are pruned. Whole frames only: a
frame is dropped only if every box in its label file is a redundant box of such a rank. A
safety margin of each rank's redundant frames is kept anyway (evenly spaced), frames that
applied runs were matched against are never dropped, and no video prefix loses more than a set
share of its frames. KEEP_REASONS lists why a frame with a redundant box can still be kept.
"""

from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
from tqdm import tqdm

from .dataset import read_label_boxes
from .paths import get_bbox_output_dir
from .redundancy import get_decisions_csv_path
from .runs import read_csv

# Why a frame containing a redundant box was kept, in the order they are checked
KEEP_REASONS = {
    "unreadable": "label file has an unparseable line",
    "kept": "has a kept (distinct) box",
    "margin": "safety margin (keep_fraction / min_keep_per_rank)",
    "rank_not_selected": "has a box from a rank not being pruned",
    "missing_image": "has a box whose image was missing during SSIM",
    "not_analysed": "has a box outside the analysed ranks",
    "reference": "is the matched kept frame of an applied run",
    "already_dropped": "already dropped by an applied run",
    "prefix_cap": "prefix cap (max_drop_fraction) reached",
}


def box_key(row):
    return (float(row["x_center"]), float(row["y_center"]), float(row["width"]), float(row["height"]))


def evenly_spaced(n_items, n_pick):
    """Indices of n_pick items spread evenly over n_items."""
    if n_pick <= 0 or n_items == 0:
        return set()
    return set(np.unique(np.linspace(0, n_items - 1, min(n_pick, n_items)).round().astype(int)).tolist())


def rank_ratio(row):
    """Distinct/count, or None for ranks without SSIM."""
    if row.get("distinct") in (None, ""):
        return None
    return int(row["distinct"]) / int(row["count"])


def plan_prune(root_dir, csv_name, rows, ranks, fingerprint, prefix_files, applied_drops,
               references, params):
    """Work out which frames to drop. Returns (manifest_rows, summary)."""
    split = rows[0]["split"]
    lbl_root = os.path.join(root_dir, "labels", split)
    decisions = read_csv(get_decisions_csv_path(os.path.join(get_bbox_output_dir(), csv_name)))
    rank_rows = {int(row["rank"]): row for row in rows}

    # Sanity check: kept rows per rank must equal the rank CSV's distinct count
    kept_counts = Counter(int(d["rank"]) for d in decisions if d["decision"] == "kept")
    for rank in ranks:
        if kept_counts[rank] != int(rank_rows[rank]["distinct"]):
            raise ValueError(f"Rank {rank}: {kept_counts[rank]} kept rows in the decisions file but "
                             f"distinct = {rank_rows[rank]['distinct']}. Analyse the prefix again without the saved results.")

    # Step 1: State of every analysed box, adding an evenly spaced safety margin of redundant boxes
    box_state = {}  # id(decision) -> state
    rank_summary = {}
    for rank in sorted({int(d["rank"]) for d in decisions}):
        rank_decisions = [d for d in decisions if int(d["rank"]) == rank]  # sorted by label = frame order
        redundant = [d for d in rank_decisions if d["decision"] == "redundant"]
        margin = set()
        if rank in ranks:
            n_margin = max(math.ceil(params["keep_fraction"] * len(redundant)), params["min_keep_per_rank"])
            margin = evenly_spaced(len(redundant), n_margin)
        for d in rank_decisions:
            if d["decision"] != "redundant":
                box_state[id(d)] = d["decision"]
            elif rank not in ranks:
                box_state[id(d)] = "rank_not_selected"
        for i, d in enumerate(redundant):
            if rank in ranks:
                box_state[id(d)] = "margin" if i in margin else "drop"
        if rank in ranks:
            rank_summary[rank] = {"redundant": len(redundant), "margin": len(margin)}

    label_decisions = defaultdict(list)
    for d in decisions:
        label_decisions[d["label"]].append(d)
    kept_boxes = {(int(d["rank"]), d["label"]): d for d in decisions if d["decision"] == "kept"}

    # Step 2: A frame is a candidate only if every box in its label file is a "drop" box
    candidates = sorted({d["label"] for d in decisions if box_state[id(d)] == "drop"})
    keep_reasons = Counter()
    frame_drops = {}  # label -> its drop decisions
    for label in tqdm(candidates, desc="Checking label files", unit="file"):
        boxes = read_label_boxes(os.path.join(lbl_root, label))
        if not boxes:
            keep_reasons["unreadable"] += 1
            continue
        available = defaultdict(list)
        for d in label_decisions[label]:
            available[box_key(d)].append(d)
        used, blocking = [], None
        for box in boxes:
            matches = available.get(box)
            if not matches:
                blocking = "not_analysed"
                break
            d = matches.pop(0)
            if box_state[id(d)] != "drop":
                blocking = box_state[id(d)]
                break
            used.append(d)
        if blocking:
            keep_reasons[blocking] += 1
            continue
        if (split, label) in references:
            keep_reasons["reference"] += 1
        elif (split, label) in applied_drops:
            keep_reasons["already_dropped"] += 1
        else:
            frame_drops[label] = used

    # Step 3: Cap the share of each prefix's frames that can be dropped (including applied runs)
    label_prefix = {}
    for prefix, files in prefix_files.items():
        for label_path in files:
            label_prefix[os.path.relpath(label_path, lbl_root)] = prefix
    already = Counter(label_prefix.get(label) for s, label in applied_drops if s == split)

    new_by_prefix = defaultdict(list)
    for label in sorted(frame_drops):
        new_by_prefix[label_prefix[label]].append(label)

    prefix_summary = {}
    for prefix, labels in new_by_prefix.items():
        before = len(prefix_files[prefix])
        allowed = max(0, math.floor(params["max_drop_fraction"] * before) - already[prefix])
        capped = 0
        if len(labels) > allowed:
            capped = len(labels) - allowed
            for i in evenly_spaced(len(labels), capped):
                del frame_drops[labels[i]]
            keep_reasons["prefix_cap"] += capped
        prefix_summary[prefix] = {"before": before, "already": already[prefix],
                                  "dropped": len(labels) - capped, "capped": capped}

    # Step 4: Manifest rows, one per box of each dropped frame
    dataset_name = rows[0]["dataset"]
    param_cols = {**params, "ssim_threshold": rows[0]["ssim_threshold"], "results_file": csv_name,
                  "results_fingerprint": fingerprint}
    manifest_rows = []
    frame_rank = {}
    for label in sorted(frame_drops):
        frame_rank[label] = min(int(d["rank"]) for d in frame_drops[label])
        for d in frame_drops[label]:
            matched = kept_boxes.get((int(d["rank"]), d["matched_label"]), {})
            manifest_rows.append({
                "dataset": dataset_name,
                "split": split,
                "video_prefix": label_prefix[label],
                "image": d["image"],
                "label": label,
                "rank": d["rank"],
                "x_center": d["x_center"],
                "y_center": d["y_center"],
                "width": d["width"],
                "height": d["height"],
                "matched_label": d["matched_label"],
                "matched_ssim": d["matched_ssim"],
                "matched_x_center": matched.get("x_center", ""),
                "matched_y_center": matched.get("y_center", ""),
                **param_cols,
            })

    # Frames are counted once, under the lowest rank they contain, so per-rank totals add up
    log_stats = Counter((label_prefix[label], rank) for label, rank in frame_rank.items())
    decision_prefix_ranks = {(label_prefix.get(d["label"]), int(d["rank"])) for d in decisions
                             if int(d["rank"]) in ranks}
    for rank in ranks:
        rank_summary[rank]["dropped"] = sum(1 for r in frame_rank.values() if r == rank)

    summary = {"ranks": rank_summary, "prefixes": prefix_summary, "keep_reasons": keep_reasons,
               "candidates": len(candidates), "dropped": len(frame_drops),
               "log_stats": log_stats, "prefix_ranks": decision_prefix_ranks,
               "prefix_before": {p: len(f) for p, f in prefix_files.items()}}
    return manifest_rows, summary


def print_prune_summary(summary, rows, split, prefix, params):
    rank_rows = {int(row["rank"]): row for row in rows}
    print("\n" + "=" * 65)
    print(f" PRUNE PLAN | SPLIT: {split.upper()} | PREFIX: {prefix}")
    print("=" * 65)
    print(f"{'Rank':<5} | {'Count':<7} | {'Distinct':<8} | {'Ratio':<6} | {'Redundant':<9} | "
          f"{'Margin':<6} | {'Frames Dropped'}")
    print("-" * 65)
    for rank, stats in sorted(summary["ranks"].items()):
        row = rank_rows[rank]
        print(f"{rank:<5} | {row['count']:<7} | {row['distinct']:<8} | {rank_ratio(row):<6.3f} | "
              f"{stats['redundant']:<9} | {stats['margin']:<6} | {stats['dropped']}")
    print("-" * 65)
    print("Frames Dropped: counted under the lowest rank the frame contains")

    if summary["prefixes"]:
        print("\n" + "-" * 65)
        print(f"{'Frames':<7} | {'Applied':<7} | {'Drop Now':<8} | {'Capped':<6} | {'% After':<7} | {'Video Prefix'}")
        print("-" * 65)
        for p, s in sorted(summary["prefixes"].items()):
            pct_after = 100 * (s["already"] + s["dropped"]) / s["before"]
            print(f"{s['before']:<7} | {s['already']:<7} | {s['dropped']:<8} | {s['capped']:<6} | "
                  f"{pct_after:<7.1f} | {p}")
        print("-" * 65)
        print("Applied: already dropped by earlier applied runs")

    print("\n" + "-" * 65)
    print(f"Frames with a box to drop:    {summary['candidates']} (redundant, outside the margin)")
    print(f"Frames to drop:               {summary['dropped']}")
    for reason, text in KEEP_REASONS.items():
        if summary["keep_reasons"][reason]:
            print(f"  kept, {text}: {summary['keep_reasons'][reason]}")
    print(f"Parameters: keep_fraction={params['keep_fraction']}, min_keep_per_rank={params['min_keep_per_rank']}, "
          f"distinct_cutoff={params['distinct_cutoff']}, max_drop_fraction={params['max_drop_fraction']}")


def make_log_rows(status, run_id, csv_name, rows, fingerprint, summary, params, manifest_name, output_dataset=""):
    """One row per (prefix, rank) the run covers."""
    rank_rows = {int(row["rank"]): row for row in rows}
    date = datetime.now().isoformat(timespec="seconds")
    log_rows = []
    for prefix, rank in sorted(summary["prefix_ranks"], key=lambda k: (str(k[0]), k[1])):
        log_rows.append({
            "date": date,
            "run_id": run_id,
            "status": status,
            "dataset": rows[0]["dataset"],
            "split": rows[0]["split"],
            "video_prefix": prefix,
            "rank": rank,
            "width": rank_rows[rank]["width"],
            "height": rank_rows[rank]["height"],
            "frames_before": summary["prefix_before"].get(prefix, ""),
            "frames_dropped": summary["log_stats"][(prefix, rank)],
            **params,
            "ssim_threshold": rows[0]["ssim_threshold"],
            "results_file": csv_name,
            "results_fingerprint": fingerprint,
            "manifest": manifest_name,
            "output_dataset": output_dataset,
        })
    return log_rows
