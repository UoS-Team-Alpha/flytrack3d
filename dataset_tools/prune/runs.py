"""The record of what has been pruned: manifests and the purge log.

A run is one set of frames to drop from one video prefix (or several). Its manifest
(purge/prune_<run_id>.csv) lists those frames, one row per box, with the kept frame each was
matched to. The purge log (purge/purge_log.csv) is append-only: a run's latest row gives its
status - planned, applied, declined, discarded, replaced or reverted. A pruned dataset is always
the original minus every applied run, so runs add up and can be replaced.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from datetime import datetime

from .dataset import get_label_files
from .paths import dataset_name, get_bbox_output_dir, get_metrics_csv_path, get_purge_log_path, get_purge_output_dir
from .redundancy import get_results_fingerprint

LOG_FIELDS = ["date", "run_id", "status", "dataset", "split", "video_prefix", "rank", "width", "height",
              "frames_before", "frames_dropped", "keep_fraction", "min_keep_per_rank", "distinct_cutoff",
              "max_drop_fraction", "ssim_threshold", "results_file", "results_fingerprint", "manifest",
              "output_dataset"]

# One row per box in a dropped frame (a frame with two boxes has two rows)
MANIFEST_FIELDS = ["run_id", "dataset", "split", "video_prefix", "image", "label", "rank",
                   "x_center", "y_center", "width", "height", "matched_label", "matched_ssim",
                   "matched_x_center", "matched_y_center", "keep_fraction", "min_keep_per_rank",
                   "distinct_cutoff", "max_drop_fraction", "ssim_threshold", "results_file",
                   "results_fingerprint"]


def read_csv(path):
    with open(path, 'r', newline='') as f:
        return list(csv.DictReader(f))


def read_purge_log(dataset):
    log_path = get_purge_log_path()
    if not os.path.exists(log_path):
        return []
    return [row for row in read_csv(log_path) if row["dataset"] == dataset]


def append_purge_log(rows):
    log_path = get_purge_log_path()
    write_header = not os.path.exists(log_path)
    with open(log_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def get_run_statuses(log_rows):
    """Latest log row per run id; its status is the run's current status."""
    runs = {}
    for row in log_rows:
        runs[row["run_id"]] = row
    return runs


def load_manifest(manifest_name):
    manifest_path = os.path.join(get_purge_output_dir(), manifest_name)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found -> {manifest_path}")
    return read_csv(manifest_path)


def get_applied_run_ids(log_rows):
    return [run_id for run_id, row in get_run_statuses(log_rows).items() if row["status"] == "applied"]


def collect_drops(log_rows, run_ids):
    """Frames dropped by the given runs' manifests, keyed by (split, label).
    Returns (drops, references): drops maps each key to its image path; references are
    the kept frames the drops were matched to, which must never be dropped themselves."""
    runs = get_run_statuses(log_rows)
    drops = {}
    references = set()
    for run_id in run_ids:
        for m in load_manifest(runs[run_id]["manifest"]):
            drops[(m["split"], m["label"])] = m["image"]
            references.add((m["split"], m["matched_label"]))
    return drops, references


def log_status_change(log_rows, run_ids, status, output_dataset=""):
    """Append a copy of each run's current rows (one per prefix/rank) with the new status."""
    runs = get_run_statuses(log_rows)
    date = datetime.now().isoformat(timespec="seconds")
    new_rows = [{**row, "date": date, "status": status, "output_dataset": output_dataset}
                for row in log_rows
                if row["run_id"] in run_ids and row["status"] == runs[row["run_id"]]["status"]]
    append_purge_log(new_rows)


def make_run_id(csv_name):
    return f"{datetime.now():%Y%m%d-%H%M%S}_{os.path.splitext(csv_name)[0][len('bbox_'):]}"


def write_manifest(run_id, manifest_rows):
    """Returns the manifest's file name (stored in the log)"""
    manifest_name = f"prune_{run_id}.csv"
    with open(os.path.join(get_purge_output_dir(), manifest_name), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows({**row, "run_id": run_id} for row in manifest_rows)
    return manifest_name


def get_prefix_runs(log_rows, split):
    """{run_id: set of video prefixes} for every currently applied run in the split."""
    runs = get_run_statuses(log_rows)
    prefix_runs = defaultdict(set)
    for row in log_rows:
        run = runs[row["run_id"]]
        if run["status"] == "applied" and row["status"] == "applied" and row["split"] == split:
            prefix_runs[row["run_id"]].add(row["video_prefix"])
    return prefix_runs


def print_purge_status(root_dir, splits, log_rows):
    """Per prefix: frames dropped by applied runs, runs only planned, and whether the bbox
    results each run used have changed since (rerun, or label files changed)"""
    runs = get_run_statuses(log_rows)
    active = {run_id: row for run_id, row in runs.items() if row["status"] in ("planned", "applied")}

    applied_frames = Counter()
    for run_id, row in active.items():
        if row["status"] != "applied":
            continue
        try:
            manifest = load_manifest(row["manifest"])
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue
        for key in {(m["split"], m["video_prefix"], m["label"]) for m in manifest}:
            applied_frames[key[:2]] += 1

    # Check each run's results against the sidecar and the current label files
    results_state = {}
    label_cache = {}
    for run_id, row in active.items():
        csv_path = os.path.join(get_bbox_output_dir(), row["results_file"])
        meta_path = os.path.splitext(csv_path)[0] + ".json"
        if not (os.path.exists(csv_path) and os.path.exists(meta_path)):
            results_state[run_id] = "missing"
            continue
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        if meta.get("fingerprint") != row["results_fingerprint"]:
            results_state[run_id] = "rerun"
            continue
        results_prefix = read_csv(csv_path)[0]["video_prefix"]
        key = (row["split"], results_prefix)
        if key not in label_cache:
            label_cache[key] = get_label_files(root_dir, *key)
        current = get_results_fingerprint(label_cache[key], meta["params"])
        results_state[run_id] = "ok" if current == row["results_fingerprint"] else "changed"

    run_prefixes = defaultdict(set)
    for row in log_rows:
        if row["run_id"] in active:
            run_prefixes[(row["split"], row["video_prefix"])].add(row["run_id"])

    # Prefixes and frame counts come from the metrics CSV, so unpurged prefixes are shown too
    prefix_frames = {}
    metrics_path = get_metrics_csv_path(root_dir)
    if os.path.exists(metrics_path):
        for row in read_csv(metrics_path):
            if row["split"] in splits and row["video_prefix"] != "TOTAL":
                prefix_frames[(row["split"], row["video_prefix"])] = int(row["frames"])
    for key in run_prefixes:
        if key[0] in splits:
            prefix_frames.setdefault(key, None)

    print("\n" + "=" * 65)
    print(f" PURGE STATUS | DATASET: {dataset_name(root_dir)}")
    print("=" * 65)
    print(f"{'Split':<6} | {'Frames':<7} | {'Dropped':<7} | {'% Drop':<6} | {'Runs':<8} | {'Results':<7} | {'Video Prefix'}")
    print("-" * 65)
    for (split, prefix), frames in sorted(prefix_frames.items()):
        dropped = applied_frames[(split, prefix)]
        run_ids = run_prefixes.get((split, prefix), set())
        statuses = {active[r]["status"] for r in run_ids}
        runs_str = "applied" if "applied" in statuses else ("planned" if statuses else "-")
        states = {results_state[r] for r in run_ids}
        results_str = "-" if not states else ("ok" if states == {"ok"} else
                                              "/".join(sorted(states - {"ok"})))
        pct = f"{100 * dropped / frames:.1f}" if frames else "-"
        print(f"{split:<6} | {frames if frames is not None else '-':<7} | {dropped:<7} | {pct:<6} | "
              f"{runs_str:<8} | {results_str:<7} | {prefix}")
    print("-" * 65)
    print("Runs:    applied (in the pruned dataset), planned (dry run only), - (never pruned)")
    print("Results: ok, rerun (bbox results saved again since), changed (label files changed),")
    print("         missing (results file deleted)")
    if not os.path.exists(metrics_path):
        print(f"Metrics CSV not found -> {metrics_path}; only prefixes with runs are shown.")
