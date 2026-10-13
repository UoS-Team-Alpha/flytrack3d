"""Prune the redundant frames of a YOLO dataset, one video prefix at a time, into a new dataset.

Pick a split in the terminal. For each of its video prefixes in turn it:
    - ranks the bbox sizes and finds the redundant boxes with SSIM (see main_analyse_redundancy.py;
      saved results are reused when the label files and settings are unchanged)
    - shows a grid of sample frames for each SSIM'd rank (close each window to continue)
    - prints the decision for each rank (prune or keep: only ranks whose distinct / count ratio
      is below distinct_cutoff are pruned) and the result (frames to drop, safety margin, cap)
    - asks y to accept, or n to keep the prefix as it is

A frame is dropped only if every box in it is redundant, so no fly is ever left unlabelled; a
safety margin of redundant frames is kept, and no video loses more than max_drop_fraction of
its frames. When every prefix is done it shows a summary and asks whether to build
output/pruned_datasets/<dataset>_pruned_<ddmmyyyy>/ (images hard-linked, labels copied,
train.txt filtered), then saves the before / after counts next to it. The original dataset is
only ever read. Train on the new folder's data.yaml in step 4.

Every answer is saved straight away: q or Ctrl+C loses nothing, and the next run offers to
continue where you left off (with the session's own settings) or to start again.

With prefixes set, only those videos are walked: a stricter second pass on the largest ones.
Their current pruning is then replaced by the new run in one rebuild (declining keeps it), and
the subset has its own session. --review opens a run's dropped frames next to the kept frame
each one matched.

The dataset and the settings come from the "guided_prune" section of the dataset pruning config.

From the IDE: edit the SETTINGS block below and press Run (or F5).
From the command line, flags override the settings and the config:
    python pipeline/step3_dataset_pruning/main_guided_prune.py
    python pipeline/step3_dataset_pruning/main_guided_prune.py --input datasets/flies --splits train val
    python pipeline/step3_dataset_pruning/main_guided_prune.py --prefixes camwidemiddle_01 widesuccess --no-grids
    python pipeline/step3_dataset_pruning/main_guided_prune.py --review-only            # view what earlier runs dropped
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # the repository root
sys.path.insert(0, str(ROOT))                # so the project packages import when this file is run directly

from common.config import load_config  # noqa: E402
from dataset_tools.prune.guided import guided_prune  # noqa: E402
from dataset_tools.prune.paths import set_output_dir  # noqa: E402
from dataset_tools.prune.review import review_prune_manifest  # noqa: E402

# ============================================================================ SETTINGS
CONFIG_PATH = ROOT / "dataset_pruning_config.json"   # "guided_prune" holds the dataset and the pruning settings
INPUT_DIR = None           # the YOLO dataset to prune (its folder or its data.yaml); None = the config's
OUTPUT_DIR = None          # where the pruned datasets are built; None = the config's (output/pruned_datasets)
SPLITS = None              # splits offered for pruning, e.g. ["train"]; None = the config's
PREFIXES = None            # only walk these videos (full names or a unique part of one); None = the config's
SHOW_GRIDS = None          # False = no rank grid windows; None = the config's
REVIEW = None              # True = afterwards, view a run's dropped frames next to their kept match; None = the config's
REVIEW_ONLY = False        # True = skip the pruning and only review the manifests of earlier runs
# =====================================================================================


def main():
    parser = argparse.ArgumentParser(description="Guided pruning of a YOLO dataset's redundant frames.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--input", default=INPUT_DIR, help="YOLO dataset to prune: its folder or its data.yaml.")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Folder the pruned datasets are built in.")
    parser.add_argument("--splits", nargs="+", default=SPLITS, help="Splits offered for pruning, e.g. train.")
    parser.add_argument("--prefixes", nargs="+", default=PREFIXES,
                        help="Only walk these videos: full video prefixes, or a part that matches exactly one.")
    parser.add_argument("--grids", action=argparse.BooleanOptionalAction, default=SHOW_GRIDS,
                        help="Show a grid of sample frames for each rank.")
    parser.add_argument("--review", action=argparse.BooleanOptionalAction, default=REVIEW,
                        help="Afterwards, view a run's dropped frames next to the kept frame each matched.")
    parser.add_argument("--review-only", action="store_true", default=REVIEW_ONLY,
                        help="Skip the pruning and only review the manifests of earlier runs.")
    args = parser.parse_args()

    if not Path(args.config).is_file():
        raise SystemExit(f"Config not found: {args.config}\n"
                         f"Copy config_examples/dataset_pruning_config.example.json to dataset_pruning_config.json "
                         f"and fill it in.")
    config = load_config(args.config)
    if "guided_prune" not in config:
        raise SystemExit("config has no 'guided_prune' section. See config_examples/dataset_pruning_config.example.json.")
    overrides = {"input_dir": args.input, "output_dir": args.output, "splits": args.splits, "prefixes": args.prefixes}
    s = {**config["guided_prune"], **{k: v for k, v in overrides.items() if v is not None}}
    grids = dict(s.get("grids") or {})
    if args.grids is not None:
        grids["enabled"] = args.grids
    review = dict(s.get("review") or {})
    if args.review is not None:
        review["enabled"] = args.review

    dataset_dir = Path(s["input_dir"]).resolve()
    if dataset_dir.is_file():   # the dataset's data.yaml was given: the dataset is its folder
        dataset_dir = dataset_dir.parent
    dataset_dir = str(dataset_dir)
    set_output_dir(s.get("output_dir") or Path(config["output_dir"]) / "pruned_datasets")

    if not args.review_only:
        guided_prune(dataset_dir, splits=tuple(s.get("splits") or ["train"]), prefixes=s.get("prefixes"),
                     top_n=int(s.get("top_n", 10)), decimals=s.get("decimals"), ssim_ranks=s.get("ssim_ranks", 3),
                     ssim_threshold=float(s.get("ssim_threshold", 0.9)),
                     keep_fraction=float(s.get("keep_fraction", 0.05)),
                     min_keep_per_rank=int(s.get("min_keep_per_rank", 10)),
                     distinct_cutoff=float(s.get("distinct_cutoff", 0.1)),
                     max_drop_fraction=float(s.get("max_drop_fraction", 0.9)),
                     show_grids=bool(grids.get("enabled", True)), grid_ranks=grids.get("ranks", "all"),
                     max_grids=grids.get("max_grids"), n=int(grids.get("num_samples", 9)),
                     grid=bool(grids.get("grid", True)))
    if args.review_only or review.get("enabled"):
        review_prune_manifest(dataset_dir, n=int(review.get("num_samples", 6)))


if __name__ == "__main__":
    main()
