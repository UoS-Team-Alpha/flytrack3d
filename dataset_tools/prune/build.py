"""Build a pruned dataset: the original minus the frames of every applied run.

The original is only ever read. Each build is a new folder, <dataset>_pruned_<ddmmyyyy>, made
as a .partial folder that is renamed only when complete. Images are hard-linked (no extra disk
space; copied if the output is on another filesystem), labels are copied, image lists
(train.txt) are filtered, and paths to the original in data.yaml are pointed at the new folder.
YOLO's labels/*.cache files are left out, so they are rebuilt from the pruned files.
"""

from __future__ import annotations

import errno
import os
import shutil
from collections import Counter
from datetime import datetime

from tqdm import tqdm

from .paths import dataset_name, get_output_dir, get_purge_log_path
from .runs import collect_drops, get_applied_run_ids, log_status_change, read_purge_log


def get_pruned_dataset_path(root_dir):
    """<output folder>/<dataset>_pruned_<ddmmyyyy> (date format matches the dataset names);
    a suffix is added if that name is taken, so an earlier pruned dataset is never overwritten."""
    base = os.path.join(get_output_dir(), f"{dataset_name(root_dir)}_pruned_{datetime.now():%d%m%Y}")
    new_root, i = base, 2
    while os.path.exists(new_root) or os.path.exists(new_root + ".partial"):
        new_root, i = f"{base}_{i}", i + 1
    return new_root


def link_or_copy(src, dst):
    """Hard link (no extra disk space); copy if the destination is on another filesystem."""
    try:
        os.link(src, dst)
        return True
    except OSError as e:
        if e.errno != errno.EXDEV:
            raise
        shutil.copy2(src, dst)
        return False


def filter_list_file(src, dst, root_dir, new_root, dropped_images):
    """Rewrite an image list (e.g. train.txt) without dropped images. Absolute paths into the
    original dataset are pointed at the new one. Returns the number of lines removed."""
    removed = 0
    with open(src, 'r') as f_in, open(dst, 'w') as f_out:
        for line in f_in:
            path = line.strip()
            rel = None
            if path:
                if os.path.isabs(path):
                    if os.path.normpath(path).startswith(root_dir + os.sep):
                        rel = os.path.relpath(path, root_dir)
                        line = line.replace(path, os.path.join(new_root, rel))
                else:
                    rel = os.path.normpath(path)
            parts = rel.split(os.sep) if rel else []
            if len(parts) > 2 and parts[0] == "images" and (parts[1], os.path.join(*parts[2:])) in dropped_images:
                removed += 1
                continue
            f_out.write(line)
    return removed


def build_pruned_dataset(root_dir, drops, references):
    """New dataset = original minus dropped frames, for every split. Built in a .partial folder
    that is renamed only when complete. The original is only ever read."""
    root_dir = os.path.normpath(os.path.abspath(root_dir))
    new_root = get_pruned_dataset_path(root_dir)
    partial = new_root + ".partial"
    if os.path.realpath(partial).startswith(os.path.realpath(root_dir) + os.sep):
        raise ValueError("Pruned dataset would be inside the original dataset.")

    # Never drop a frame that an applied run matched its drops against
    drops = {key: image for key, image in drops.items() if key not in references}
    dropped_labels = set(drops)
    dropped_images = {(split, image) for (split, _), image in drops.items()}

    tasks = []
    for sub in ("images", "labels"):
        for dirpath, _, filenames in os.walk(os.path.join(root_dir, sub)):
            rel_dir = os.path.relpath(dirpath, root_dir)
            os.makedirs(os.path.join(partial, rel_dir), exist_ok=True)
            for name in filenames:
                tasks.append((sub, rel_dir, name))

    stats = Counter()
    for sub, rel_dir, name in tqdm(tasks, desc="Building pruned dataset", unit="file"):
        rel = os.path.join(rel_dir, name)
        parts = rel.split(os.sep)
        key = (parts[1], os.path.join(*parts[2:])) if len(parts) > 2 else None
        if sub == "labels" and name.endswith(".cache"):
            stats["cache_skipped"] += 1  # YOLO cache would still list dropped frames
            continue
        if key in (dropped_images if sub == "images" else dropped_labels):
            stats[f"{sub}_dropped"] += 1
            continue
        src, dst = os.path.join(root_dir, rel), os.path.join(partial, rel)
        if sub == "images":
            stats["images_linked" if link_or_copy(src, dst) else "images_copied"] += 1
        else:
            shutil.copy2(src, dst)  # copied so later label edits can't change the original
            stats["labels_copied"] += 1

    # Root-level files: filter image lists, copy the rest (data.yaml)
    list_removed = {}
    for name in sorted(os.listdir(root_dir)):
        src, dst = os.path.join(root_dir, name), os.path.join(partial, name)
        if os.path.isdir(src):
            if name not in ("images", "labels"):
                print(f"Warning: folder '{name}' was not copied to the pruned dataset.")
        elif name.endswith(".txt"):
            list_removed[name] = filter_list_file(src, dst, root_dir, new_root, dropped_images)
        elif name.endswith(".cache"):
            stats["cache_skipped"] += 1
        elif name.endswith((".yaml", ".yml")):
            with open(src, 'r') as f:
                text = f.read()
            if root_dir in text:
                text = text.replace(root_dir, new_root)
                print(f"Note: paths to the original dataset in {name} now point at the pruned dataset.")
            # A relative 'path:' is resolved by Ultralytics against its datasets folder, not this
            # file's folder, so it may still point at the original dataset
            for line in text.splitlines():
                key, _, value = line.partition(":")
                value = value.split("#")[0].strip().strip("'\"")
                if key.strip() == "path" and value not in ("", ".") and not os.path.isabs(value):
                    print(f"Warning: {name} has a relative 'path: {value}'. Check it points at the "
                          f"pruned dataset, or remove it to use the yaml's folder.")
            with open(dst, 'w') as f:
                f.write(text)
            shutil.copystat(src, dst)
        else:
            shutil.copy2(src, dst)

    # Every dropped frame must have been found, or the manifests don't match this dataset
    if stats["images_dropped"] != len(dropped_images) or stats["labels_dropped"] != len(dropped_labels):
        raise RuntimeError(f"Expected to drop {len(dropped_images)} images / {len(dropped_labels)} labels, "
                           f"found {stats['images_dropped']} / {stats['labels_dropped']}. "
                           f"Incomplete dataset left in {partial}")

    os.rename(partial, new_root)

    print("\n" + "-" * 65)
    print(f"Images hard-linked:   {stats['images_linked']}")
    if stats["images_copied"]:
        print(f"Images copied:        {stats['images_copied']} (different filesystem)")
    print(f"Labels copied:        {stats['labels_copied']}")
    print(f"Frames dropped:       {stats['images_dropped']}")
    for name, removed in list_removed.items():
        print(f"Removed from {name}: {removed}")
    if stats["cache_skipped"]:
        print(f"Cache files skipped:  {stats['cache_skipped']} (YOLO rebuilds them)")
    print(f"Pruned dataset: {new_root}")
    return new_root


def apply_runs(root_dir, run_ids, replace_run_ids=()):
    """Build a pruned dataset from the original minus every applied run plus these planned
    runs, then log these runs as applied. Runs in replace_run_ids are left out of the build
    and logged as replaced. Returns the new dataset's path."""
    log_rows = read_purge_log(dataset_name(root_dir))
    applied = [r for r in get_applied_run_ids(log_rows) if r not in replace_run_ids]
    drops, references = collect_drops(log_rows, applied + list(run_ids))
    new_root = build_pruned_dataset(root_dir, drops, references)
    log_status_change(log_rows, run_ids, "applied", output_dataset=new_root)
    if replace_run_ids:
        log_status_change(log_rows, replace_run_ids, "replaced", output_dataset=new_root)
        print(f"Logged {len(replace_run_ids)} earlier run(s) as replaced.")
    print(f"Logged as applied: {get_purge_log_path()}")
    return new_root
