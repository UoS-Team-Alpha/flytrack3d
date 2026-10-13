"""Find the redundant frames of a YOLO dataset: boxes of the same size that also look the same.

The dataset was built from tracked video, so a fly that sits still gets the same bounding box
size in hundreds of consecutive frames. Pick a split and a video prefix (or all prefixes) in
the terminal. Its most common bbox sizes are ranked (rank 1 = the most common), and for the top
ranks every box is cropped and compared with SSIM in frame order: a crop is kept if it differs
from every crop kept so far, else it is redundant. A rank's distinct / count ratio says how
varied it is: a still fly might give 12 distinct crops out of 782. Then pick saved results and
a rank to see sample frames of it with the boxes drawn (close the window for another rank).

Nothing here changes the dataset; it is the look-before-you-prune half of this step, and
main_guided_prune.py runs the same analysis itself. The dataset and the settings come from the
"analyse_redundancy" section of the dataset pruning config. Results are saved to
output/pruned_datasets/bbox_sizes/ and reused while the label files and settings are unchanged.
The split and prefix tables come from output/pruned_datasets/<dataset>_metrics.csv, which is
made on the first run (--refresh-metrics to count the dataset again).

All prompts take a number; q quits and Ctrl+C exits cleanly.

From the IDE: edit the SETTINGS block below and press Run (or F5).
From the command line, flags override the settings and the config:
    python pipeline/step3_dataset_pruning/main_analyse_redundancy.py
    python pipeline/step3_dataset_pruning/main_analyse_redundancy.py --input datasets/flies --no-grids
    python pipeline/step3_dataset_pruning/main_analyse_redundancy.py --grids-only        # view results saved earlier
    python pipeline/step3_dataset_pruning/main_analyse_redundancy.py --refresh-metrics
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # the repository root
sys.path.insert(0, str(ROOT))                # so the project packages import when this file is run directly

from common.config import load_config  # noqa: E402
from dataset_tools.prune.dataset import check_layout  # noqa: E402
from dataset_tools.prune.metrics import write_video_metrics  # noqa: E402
from dataset_tools.prune.paths import get_metrics_csv_path, set_output_dir  # noqa: E402
from dataset_tools.prune.redundancy import analyse_bbox_duplicates  # noqa: E402
from dataset_tools.prune.review import view_rank_grids  # noqa: E402

# ============================================================================ SETTINGS
CONFIG_PATH = ROOT / "configs" / "dataset_pruning.json"   # "analyse_redundancy" holds the dataset and the settings
INPUT_DIR = None           # the YOLO dataset to analyse (its folder or its data.yaml); None = the config's
OUTPUT_DIR = None          # where the results are kept; None = the config's (output/pruned_datasets)
REFRESH_METRICS = None     # True = count the dataset again instead of reading the saved metrics CSV; None = the config's
SHOW_GRIDS = None          # False = skip the rank grids after the analysis; None = the config's
GRIDS_ONLY = False         # True = skip the analysis and only view the rank grids of saved results
# =====================================================================================


def main():
    parser = argparse.ArgumentParser(description="Rank a dataset's bbox sizes and find its redundant boxes with SSIM.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--input", default=INPUT_DIR, help="YOLO dataset to analyse: its folder or its data.yaml.")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Folder the results are kept in.")
    parser.add_argument("--refresh-metrics", action=argparse.BooleanOptionalAction, default=REFRESH_METRICS,
                        help="Count the dataset again instead of reading the saved metrics CSV.")
    parser.add_argument("--grids", action=argparse.BooleanOptionalAction, default=SHOW_GRIDS,
                        help="View sample frames of a rank after the analysis.")
    parser.add_argument("--grids-only", action="store_true", default=GRIDS_ONLY,
                        help="Skip the analysis and only view the rank grids of saved results.")
    args = parser.parse_args()

    if not Path(args.config).is_file():
        raise SystemExit(f"Config not found: {args.config}\n"
                         f"Copy configs/dataset_pruning.example.json to configs/dataset_pruning.json "
                         f"and fill it in.")
    config = load_config(args.config)
    if "analyse_redundancy" not in config:
        raise SystemExit("config has no 'analyse_redundancy' section. "
                         "See configs/dataset_pruning.example.json.")
    overrides = {"input_dir": args.input, "output_dir": args.output, "refresh_metrics": args.refresh_metrics}
    s = {**config["analyse_redundancy"], **{k: v for k, v in overrides.items() if v is not None}}
    grids = dict(s.get("grids") or {})
    if args.grids is not None:
        grids["enabled"] = args.grids

    dataset_dir = Path(s["input_dir"]).resolve()
    if dataset_dir.is_file():   # the dataset's data.yaml was given: the dataset is its folder
        dataset_dir = dataset_dir.parent
    dataset_dir = str(dataset_dir)
    check_layout(dataset_dir)
    set_output_dir(s.get("output_dir") or Path(config["output_dir"]) / "pruned_datasets")

    try:
        if not args.grids_only:
            if s.get("refresh_metrics") or not Path(get_metrics_csv_path(dataset_dir)).is_file():
                write_video_metrics(dataset_dir)
            # ssim and save write the decisions file that guided pruning and the grids reuse
            analyse_bbox_duplicates(dataset_dir, top_n=int(s.get("top_n", 10)), decimals=s.get("decimals"),
                                    save_csv=bool(s.get("save", True)), ssim=bool(s.get("ssim", True)),
                                    ssim_threshold=float(s.get("ssim_threshold", 0.9)), ssim_ranks=s.get("ssim_ranks"),
                                    use_cache=bool(s.get("use_cache", True)))
    except (KeyboardInterrupt, EOFError):
        raise SystemExit("\nExiting.")
    if args.grids_only or grids.get("enabled", True):
        view_rank_grids(dataset_dir, n=int(grids.get("num_samples", 9)), grid=bool(grids.get("grid", True)))


if __name__ == "__main__":
    main()
