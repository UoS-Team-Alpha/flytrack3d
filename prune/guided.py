"""Guided pruning: walk a whole split one video prefix at a time and build one pruned dataset.

For each video prefix it analyses the bbox sizes with SSIM (redundancy.py), shows a grid per
SSIM'd rank, proposes which ranks to prune and what that drops (plan.py), and asks y/n. Each
answer is saved to a session file straight away, so a run stopped with q or Ctrl+C can be
continued later. When every prefix is done, the accepted drops are built into one new pruned
dataset (build.py); the original is never modified.

With prefixes=[...] only those videos are walked (e.g. a stricter second pass on the largest
ones). A video's existing applied run is then replaced by the new one rather than stacked.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt

from .build import apply_runs
from .dataset import check_layout, count_dataset, get_label_files, group_by_prefix
from .metrics import load_metrics_rows, prompt_split, write_video_metrics
from .paths import get_metrics_csv_path, get_output_dir, get_purge_output_dir
from .plan import make_log_rows, plan_prune, print_prune_summary, rank_ratio
from .prompts import prompt_choice, prompt_yes_no
from .redundancy import get_results_fingerprint, run_bbox_analysis
from .review import read_label_bboxes, show_rank_grid
from .runs import (append_purge_log, collect_drops, get_applied_run_ids, get_prefix_runs, get_run_statuses,
                   log_status_change, make_run_id, print_purge_status, read_csv, read_purge_log, write_manifest)


class QuitGuided(Exception):
    """Raised when the user types q at any prompt."""


def resolve_prefixes(requested, available):
    """Match each requested name to a video prefix: an exact name, or a part of exactly one name."""
    resolved = []
    for name in requested:
        if name in available:
            resolved.append(name)
            continue
        matches = [p for p in available if name in p]
        if len(matches) != 1:
            problem = "matches no video prefix" if not matches else f"matches {len(matches)} prefixes"
            listing = "".join(f"\n  {p}" for p in matches[:10])
            raise ValueError(f"'{name}' {problem}. Use a longer part of the name.{listing}")
        resolved.append(matches[0])
    return list(dict.fromkeys(resolved))


def get_session_path(dataset_name, split, subset=None):
    """A subset of prefixes gets its own session, so it never disturbs the full-split one."""
    if not subset:
        return os.path.join(get_purge_output_dir(), f"guided_{split}_{dataset_name}.json")
    subset_hash = hashlib.sha1("\n".join(sorted(subset)).encode()).hexdigest()[:6]
    return os.path.join(get_purge_output_dir(), f"guided_{split}_{dataset_name}_subset_{subset_hash}.json")


def load_session(session_path):
    if not os.path.exists(session_path):
        return None
    with open(session_path, 'r') as f:
        return json.load(f)


def save_session(session_path, session):
    """Write to a temporary file first, so an interrupted save never corrupts the session."""
    session["updated"] = datetime.now().isoformat(timespec="seconds")
    tmp_path = session_path + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(session, f, indent=2)
    os.replace(tmp_path, session_path)


def session_label(session):
    subset = session.get("subset")
    return f"{session['split'].upper()}" + (f" | {len(subset)} SELECTED PREFIXES" if subset else "")


def print_session_progress(session):
    done = session["done"]
    outcomes = [d["outcome"] for d in done.values()]
    accepted = [d for d in done.values() if d["outcome"] == "accepted"]
    print("\n" + "=" * 65)
    print(f" GUIDED SESSION | SPLIT: {session_label(session)} | STARTED: {session['created']}")
    print("=" * 65)
    print(f"Prefixes done:        {len(done)} / {len(session['prefixes'])}")
    print(f"Accepted:             {len(accepted)} ({sum(d['frames_dropped'] for d in accepted)} frames to drop)")
    print(f"Declined:             {outcomes.count('declined')}")
    print(f"Nothing to prune:     {len(outcomes) - len(accepted) - outcomes.count('declined')}")
    print(f"Last updated:         {session['updated']}")


def discard_session(session, dataset_name):
    """Accepted runs that were never built are marked discarded, so they stop showing as planned."""
    run_ids = [d["run_id"] for d in session["done"].values() if d["outcome"] == "accepted"]
    log_rows = read_purge_log(dataset_name)
    runs = get_run_statuses(log_rows)
    run_ids = [r for r in run_ids if r in runs and runs[r]["status"] == "planned"]
    if run_ids:
        log_status_change(log_rows, run_ids, "discarded")


def start_or_resume_session(session_path, dataset_name, split, subset, analysis_params, prune_params):
    """Returns the session to use: the saved one (continue) or a new one (start again),
    or None if the last session was built and the user doesn't want a new one."""
    session = load_session(session_path)
    what = f"'{split}'" + (f" ({len(subset)} selected prefixes)" if subset else "")

    if session and session.get("built"):
        print(f"\nThe last guided session for {what} was built into:\n  {session['built']}")
        answer = prompt_yes_no("Start a new guided session? Applied drops are kept (y/n, q to quit): ")
        if answer is None:
            raise QuitGuided
        if not answer:
            return None
        session = None

    elif session:
        print_session_progress(session)
        options = ["Continue where you left off", "Start again (accepted, unbuilt prefixes are discarded)"]
        print("-" * 65)
        for i, option in enumerate(options, 1):
            print(f"{i:<4} | {option}")
        choice = prompt_choice(options, "\nSelect number (q to quit): ")
        if choice is None:
            raise QuitGuided
        if choice == options[1]:
            discard_session(session, dataset_name)
            session = None
        elif (session["analysis_params"], session["prune_params"]) != (analysis_params, prune_params):
            print("\nContinuing with the session's own parameters (they differ from those passed in):")
            print(f"  {session['analysis_params']}\n  {session['prune_params']}")

    if session is None:
        session = {
            "dataset": dataset_name,
            "split": split,
            "subset": subset,
            "created": datetime.now().isoformat(timespec="seconds"),
            "analysis_params": analysis_params,
            "prune_params": prune_params,
            "prefixes": [],
            "done": {},
            "built": None,
        }
    return session


def get_replaceable_runs(log_rows, split, prefix):
    """Applied runs that cover only this prefix (e.g. its guided run) can be replaced by a new
    run. Runs that also cover other prefixes can't be replaced for just one of them.
    Returns (replaceable, shared) lists of run ids."""
    replaceable, shared = [], []
    for run_id, prefixes in get_prefix_runs(log_rows, split).items():
        if prefix in prefixes:
            (replaceable if prefixes == {prefix} else shared).append(run_id)
    return replaceable, shared


def print_rank_decisions(ssim_rows, distinct_cutoff):
    """The outcome: which SSIM'd ranks will be pruned and why. Returns the ranks to prune."""
    print("\n" + "=" * 65)
    print(" PROPOSED RANKS TO PRUNE")
    print("=" * 65)
    print(f"{'Rank':<5} | {'Width':<10} | {'Height':<10} | {'Count':<7} | {'Distinct':<8} | {'Ratio':<6} | {'Decision'}")
    print("-" * 65)
    ranks = []
    for row in ssim_rows:
        ratio = rank_ratio(row)
        prune = ratio < distinct_cutoff
        if prune:
            ranks.append(int(row["rank"]))
        print(f"{row['rank']:<5} | {row['width']:<10} | {row['height']:<10} | {row['count']:<7} | "
              f"{row['distinct']:<8} | {ratio:<6.3f} | {'prune' if prune else 'keep'}")
    print("-" * 65)
    print(f"Ratio: distinct/count. Ranks below {distinct_cutoff} are mostly repeats of the same")
    print("appearance, so their redundant frames are pruned.")
    return ranks


def process_prefix(root_dir, split, prefix, prefix_files, analysis_params, prune_params, n, grid,
                   show_grids, grid_ranks, max_grids, replace):
    """Analyse, visualise and propose drops for one video prefix. Returns its session entry.
    replace=True plans as if this prefix's own applied runs were not there, so accepting
    replaces them (declining keeps them)."""
    dataset_name = os.path.basename(os.path.normpath(root_dir))
    date = datetime.now().isoformat(timespec="seconds")
    log_rows = read_purge_log(dataset_name)

    replace_ids, shared_ids = get_replaceable_runs(log_rows, split, prefix) if replace else ([], [])
    current = sum(int(r["frames_dropped"]) for r in log_rows
                  if r["run_id"] in replace_ids and r["status"] == "applied")
    if replace_ids:
        print(f"Currently pruned by {len(replace_ids)} applied run(s): {current} frames dropped. "
              f"Accepting replaces this; declining keeps it.")
    if shared_ids:
        print(f"Note: {len(shared_ids)} applied run(s) also cover other prefixes and are kept; "
              f"new drops are added on top of them.")
    kept_text = "Its current pruning is kept." if replace_ids else "All frames of this prefix are kept."

    # Step 1: bbox analysis with SSIM (reuses saved results if labels and parameters are unchanged)
    result = run_bbox_analysis(root_dir, split, prefix, save_csv=True, use_cache=True,
                               label_files=prefix_files[prefix], **analysis_params)
    if result is None:
        return {"outcome": "no_bboxes", "frames_dropped": 0, "date": date}
    csv_name = os.path.basename(result["csv_path"])
    rows = read_csv(result["csv_path"])
    ssim_rows = [row for row in rows if rank_ratio(row) is not None]

    # Step 2: Grids for the SSIM'd ranks (all, or only those that will be pruned, up to max_grids)
    if show_grids:
        grid_rows = ssim_rows
        if grid_ranks == "pruned":
            grid_rows = [row for row in ssim_rows if rank_ratio(row) < prune_params["distinct_cutoff"]]
        grid_rows = grid_rows[:max_grids] if max_grids is not None else grid_rows
        if len(grid_rows) < len(ssim_rows):
            print(f"\nShowing grids for {len(grid_rows)} of {len(ssim_rows)} SSIM'd ranks "
                  f"(grid_ranks='{grid_ranks}', max_grids={max_grids}).")
        if grid_rows:
            label_bboxes = read_label_bboxes(prefix_files[prefix])
            for row in grid_rows:
                show_rank_grid(label_bboxes, row, root_dir, n, grid, close_text="continue")

    # Step 3: The outcome (ranks to prune), then the result (frames that would be dropped)
    ranks = print_rank_decisions(ssim_rows, prune_params["distinct_cutoff"])
    if not ranks:
        print(f"\nNo ranks below the cut-off. {kept_text}")
        return {"outcome": "nothing_to_prune", "frames_dropped": 0, "date": date}

    applied = [r for r in get_applied_run_ids(log_rows) if r not in replace_ids]
    applied_drops, references = collect_drops(log_rows, applied)
    manifest_rows, summary = plan_prune(root_dir, csv_name, rows, ranks, result["fingerprint"], prefix_files,
                                        applied_drops, references, prune_params)
    print_prune_summary(summary, rows, split, prefix, prune_params)
    if not manifest_rows:
        print(f"\nNothing to drop after the safety margin, cap and existing drops. {kept_text}")
        return {"outcome": "nothing_to_drop", "frames_dropped": 0, "date": date}

    # Step 4: Ask the user to agree
    question = (f"\nReplace the current {current} dropped frames with these {summary['dropped']}? "
                if replace_ids else f"\nDrop these {summary['dropped']} frames of {len(prefix_files[prefix])}? ")
    answer = prompt_yes_no(question + "(y/n, q to quit): ")
    if answer is None:
        raise QuitGuided

    run_id = make_run_id(csv_name)
    entry = {"run_id": run_id, "frames_dropped": summary["dropped"], "results_file": csv_name,
             "results_fingerprint": result["fingerprint"], "date": date}
    if answer:
        manifest_name = write_manifest(run_id, manifest_rows)
        append_purge_log(make_log_rows("planned", run_id, csv_name, rows, result["fingerprint"], summary,
                                       prune_params, manifest_name))
        print(f"Accepted. Manifest saved: {manifest_name}")
        return {**entry, "outcome": "accepted", "manifest": manifest_name, "replaces": replace_ids}

    append_purge_log(make_log_rows("declined", run_id, csv_name, rows, result["fingerprint"], summary,
                                   prune_params, ""))
    print(f"Declined. {kept_text}")
    return {**entry, "outcome": "declined"}


def finish_session(root_dir, session, session_path, prefix_files):
    """Every prefix is done: summarise, check the accepted results are still current, then build."""
    dataset_name = session["dataset"]
    done = session["done"]

    print("\n" + "=" * 65)
    print(f" GUIDED SESSION COMPLETE | SPLIT: {session_label(session)}")
    print("=" * 65)
    print(f"{'Outcome':<16} | {'Frames':<7} | {'Dropped':<7} | {'Video Prefix'}")
    print("-" * 65)
    for prefix in session["prefixes"]:
        if prefix in done:
            d = done[prefix]
            print(f"{d['outcome']:<16} | {len(prefix_files.get(prefix, [])):<7} | {d['frames_dropped']:<7} | {prefix}")
    print("-" * 65)

    accepted = {p: d for p, d in done.items() if d["outcome"] == "accepted"}
    if not accepted:
        print("No prefixes were accepted, so there is nothing to build.")
        session["built"] = ""
        save_session(session_path, session)
        return

    # Label files changed since a prefix was accepted -> its drops may be wrong, so redo it
    stale = [p for p, d in accepted.items()
             if get_results_fingerprint(prefix_files.get(p, []), session["analysis_params"]) != d["results_fingerprint"]]
    if stale:
        print(f"The label files of {len(stale)} accepted prefix(es) changed since they were accepted:")
        for p in stale:
            print(f"  {p}")
        log_status_change(read_purge_log(dataset_name), [accepted[p]["run_id"] for p in stale], "discarded")
        for p in stale:
            del done[p]
        save_session(session_path, session)
        print("They have been reset. Run guided pruning again and continue to redo them.")
        return

    # Runs to replace must still be applied (one could have been reverted in the meantime)
    still_applied = set(get_applied_run_ids(read_purge_log(dataset_name)))
    replace_ids = [r for d in accepted.values() for r in d.get("replaces", []) if r in still_applied]

    total = sum(d["frames_dropped"] for d in accepted.values())
    print(f"Accepted: {len(accepted)} prefix(es), {total} frames to drop.")
    if replace_ids:
        print(f"These replace {len(replace_ids)} earlier applied run(s) for the same prefixes.")
    print("Building makes a new dataset folder (all splits) from the original minus these drops and")
    print("any other applied runs. The original is not modified.")
    answer = prompt_yes_no("Build the pruned dataset now? (y/n, q to quit): ")
    if not answer:
        print("Not built. Run guided pruning again and continue to build it.")
        return

    earlier_runs = [r for r in still_applied if r not in replace_ids]
    new_root = apply_runs(root_dir, [d["run_id"] for d in accepted.values()], replace_run_ids=replace_ids)
    session["built"] = new_root
    save_session(session_path, session)
    write_guided_summary(root_dir, new_root, session, earlier_runs, replace_ids)


def pct_dropped(before, after):
    return f"{100 * (before - after) / before:.1f}" if before else "-"


def write_guided_summary(root_dir, new_root, session, earlier_runs, replaced_runs=()):
    """Before/after stats for the finished guided prune, printed and saved to
    <output folder>/<new dataset>_summary_<date-time>.txt."""
    print("\nCounting images and instances in the original and new datasets...")
    before, after = count_dataset(root_dir), count_dataset(new_root)
    split = session["split"]
    now = datetime.now()
    done = session["done"]
    outcomes = [d["outcome"] for d in done.values()]
    params = {**session["analysis_params"], **session["prune_params"]}
    not_done = "not in session" if session.get("subset") else "not reviewed"

    lines = [
        "=" * 65,
        " GUIDED PRUNE SUMMARY",
        "=" * 65,
        f"Date:               {now:%Y-%m-%d %H:%M:%S}",
        f"Original dataset:   {os.path.basename(os.path.normpath(root_dir))}",
        f"                    {os.path.normpath(root_dir)}",
        f"New dataset:        {os.path.basename(new_root)}",
        f"                    {new_root}",
        f"Split pruned:       {split}" + (f" ({len(session['subset'])} selected prefixes)" if session.get("subset") else ""),
        f"Session started:    {session['created']}",
        f"Parameters:         " + ", ".join(f"{k}={v}" for k, v in params.items()),
        f"Prefixes:           {len(done)} reviewed | {outcomes.count('accepted')} accepted | "
        f"{outcomes.count('declined')} declined | "
        f"{len(outcomes) - outcomes.count('accepted') - outcomes.count('declined')} nothing to prune",
        f"Earlier runs:       {len(earlier_runs)} applied run(s) from before this session also included",
    ]
    if replaced_runs:
        lines.append(f"Replaced runs:      {len(replaced_runs)} earlier run(s) replaced by this session")
    lines += [
        "",
        "-" * 65,
        " DATASET TOTALS (all splits, compared with the original)",
        "-" * 65,
        f"{'Split':<6} | {'Images':<15} | {'% Drop':<6} | {'Instances':<15} | {'% Drop'}",
        "-" * 65,
    ]
    for s in sorted(set(before) | set(after)):
        b, a = before.get(s, {}), after.get(s, {})
        bi, ai = b.get("images", 0), a.get("images", 0)
        bn, an = b.get("instances", 0), a.get("instances", 0)
        lines.append(f"{s:<6} | {f'{bi} -> {ai}':<15} | {pct_dropped(bi, ai):<6} | "
                     f"{f'{bn} -> {an}':<15} | {pct_dropped(bn, an)}")
    bi, ai = sum(v["images"] for v in before.values()), sum(v["images"] for v in after.values())
    bn, an = sum(v["instances"] for v in before.values()), sum(v["instances"] for v in after.values())
    lines += [
        "-" * 65,
        f"{'TOTAL':<6} | {f'{bi} -> {ai}':<15} | {pct_dropped(bi, ai):<6} | "
        f"{f'{bn} -> {an}':<15} | {pct_dropped(bn, an)}",
        "",
        "-" * 65,
        f" PER VIDEO PREFIX | SPLIT: {split.upper()}",
        "-" * 65,
        f"{'Frames':<15} | {'% Drop':<6} | {'Instances':<15} | {'% Drop':<6} | {'Outcome':<16} | {'Video Prefix'}",
        "-" * 65,
    ]
    for prefix, b in sorted(before[split]["prefixes"].items()):
        a = after.get(split, {}).get("prefixes", {}).get(prefix, {"frames": 0, "instances": 0})
        outcome = done.get(prefix, {}).get("outcome", not_done)
        frames = f"{b['frames']} -> {a['frames']}"
        instances = f"{b['instances']} -> {a['instances']}"
        lines.append(f"{frames:<15} | {pct_dropped(b['frames'], a['frames']):<6} | {instances:<15} | "
                     f"{pct_dropped(b['instances'], a['instances']):<6} | {outcome:<16} | {prefix}")
    lines += ["-" * 65,
              "Frames: label files. Instances: box lines. Other splits are copied unchanged.",
              "Counts include every applied run, so prefixes outside this session show earlier pruning."]

    summary_path = os.path.join(get_output_dir(),
                                f"{os.path.basename(new_root)}_summary_{now:%Y%m%d-%H%M%S}.txt")
    with open(summary_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nSummary saved to: {summary_path}")


def guided_prune(root_dir, splits=("train",), top_n=10, ssim=True, ssim_ranks=3, ssim_threshold=0.9,
                 decimals=None, keep_fraction=0.05, min_keep_per_rank=10, distinct_cutoff=0.1,
                 max_drop_fraction=0.9, n=9, grid=True, show_grids=True, prefixes=None,
                 grid_ranks="all", max_grids=None):
    """Walk a split one video prefix at a time and build one pruned dataset at the end.
    q at any prompt or Ctrl+C (in the terminal or while a window is open) exits cleanly.
    Progress is saved after every prefix, so running again offers to continue.
    prefixes: only walk these videos (full names or a unique part of one); their existing
              applied runs are replaced by the new ones when accepted.
    grid_ranks: "all" SSIM'd ranks, or only those that will be "pruned"; max_grids caps the count."""
    try:
        _guided_prune(root_dir, splits, top_n, ssim, ssim_ranks, ssim_threshold, decimals, keep_fraction,
                      min_keep_per_rank, distinct_cutoff, max_drop_fraction, n, grid, show_grids,
                      prefixes, grid_ranks, max_grids)
    except (QuitGuided, KeyboardInterrupt, EOFError):
        print("\nExiting. Progress is saved; run guided pruning again to continue.")
    finally:
        plt.close('all')


def _guided_prune(root_dir, splits, top_n, ssim, ssim_ranks, ssim_threshold, decimals, keep_fraction,
                  min_keep_per_rank, distinct_cutoff, max_drop_fraction, n, grid, show_grids,
                  prefixes, grid_ranks, max_grids):
    if not ssim:
        print("Guided pruning needs ssim=True: frames are only dropped using the SSIM decisions.")
        return
    if grid_ranks not in ("all", "pruned"):
        print(f"grid_ranks must be 'all' or 'pruned', not '{grid_ranks}'.")
        return
    check_layout(root_dir)
    dataset_name = os.path.basename(os.path.normpath(root_dir))
    analysis_params = {"top_n": top_n, "decimals": decimals, "ssim": ssim,
                       "ssim_threshold": ssim_threshold, "ssim_ranks": ssim_ranks}
    prune_params = {"keep_fraction": keep_fraction, "min_keep_per_rank": min_keep_per_rank,
                    "distinct_cutoff": distinct_cutoff, "max_drop_fraction": max_drop_fraction}

    # Step 1: Select the split. The metrics CSV only feeds the split and status tables, so
    # if it is missing it is created here (for every split, so no rows are lost)
    if not os.path.exists(get_metrics_csv_path(root_dir)):
        print("Metrics CSV not found; counting the dataset first.")
        write_video_metrics(root_dir)
    metrics_rows = load_metrics_rows(root_dir)
    if metrics_rows is None:
        return
    print_purge_status(root_dir, splits, read_purge_log(dataset_name))
    split = prompt_split(metrics_rows, splits)
    if split is None:
        return

    # Step 2: Group the split's label files by video prefix (one walk for the whole split),
    # and resolve the selected prefixes if a subset was asked for
    print(f"\nReading the '{split}' label files...")
    lbl_root = os.path.join(root_dir, "labels", split)
    prefix_files = group_by_prefix(get_label_files(root_dir, split), lbl_root)
    subset = None
    if prefixes:
        try:
            subset = resolve_prefixes(prefixes, sorted(prefix_files))
        except ValueError as e:
            print(e)
            return
        print(f"Selected {len(subset)} prefix(es):")
        for p in subset:
            print(f"  {len(prefix_files[p]):<6} frames | {p}")

    # Step 3: Continue the saved session or start again
    session_path = get_session_path(dataset_name, split, subset)
    session = start_or_resume_session(session_path, dataset_name, split, subset, analysis_params, prune_params)
    if session is None:
        print("Nothing to do.")
        return

    if subset:
        session["prefixes"] = subset
    else:
        new_prefixes = [p for p in sorted(prefix_files) if p not in session["prefixes"]]
        if session["prefixes"] and new_prefixes:
            print(f"{len(new_prefixes)} new prefix(es) found since the session started; they are added at the end.")
        session["prefixes"] += new_prefixes
    missing = [p for p in session["prefixes"] if p not in prefix_files]
    if missing:
        print(f"Warning: {len(missing)} prefix(es) in the session no longer exist in the split and are skipped.")
    save_session(session_path, session)

    # Step 4: Each remaining prefix in turn
    remaining = [p for p in session["prefixes"] if p not in session["done"] and p in prefix_files]
    total = len(session["prefixes"])
    for prefix in remaining:
        number = session["prefixes"].index(prefix) + 1
        print("\n" + "#" * 65)
        print(f" VIDEO PREFIX {number}/{total}: {prefix} ({len(prefix_files[prefix])} frames)")
        print("#" * 65)
        session["done"][prefix] = process_prefix(root_dir, split, prefix, prefix_files,
                                                 session["analysis_params"], session["prune_params"],
                                                 n, grid, show_grids, grid_ranks, max_grids,
                                                 replace=bool(session.get("subset")))
        save_session(session_path, session)

    # Step 5: Build the pruned dataset from everything accepted
    finish_session(root_dir, session, session_path, prefix_files)
