"""Report what a YOLO dataset holds: its splits, the videos in each, and its bounding boxes.

For each split (train / val / test) it prints the video prefixes with their image and
instance counts, the background images, the image sizes and the distribution of the box
sizes and of the instances per image. It also warns about what is usually a mistake: a video
that is in more than one split, images without a label file, label files without an image and
boxes reaching outside their image. Two figures are then shown:

    contents  pie charts of the images and instances per split, and per video within each split
    boxes     per split, where the boxes sit in the frame, their width against their height
              and a histogram of their size

A frame's video is the part of its file name before "_frame_" (prefix_separator), else its
sub-folder of the split. The dataset comes from the "analyse" section of the image dataset
config and is never changed. The report (report.txt, summary.json) and the figures
(contents.png, boxes.png) are saved to output/dataset_analysis/<dataset name>/ unless
output_dir is set. With samples enabled, a few random images are also opened with their
boxes drawn, as after tiling (press any key for the next one).

From the IDE: edit the SETTINGS block below and press Run (or F5).
From the command line, flags override the settings and the config:
    python pipeline/step1_image_dataset_creation/main_analyse_dataset.py
    python pipeline/step1_image_dataset_creation/main_analyse_dataset.py --input datasets/flies --splits train val
    python pipeline/step1_image_dataset_creation/main_analyse_dataset.py --no-show          # report and saved figures only
    python pipeline/step1_image_dataset_creation/main_analyse_dataset.py --samples --no-save
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # the repository root
sys.path.insert(0, str(ROOT))                # so the project packages import when this file is run directly

from common.config import load_config  # noqa: E402
from common.datasets import read_class_names  # noqa: E402
from dataset_tools.stats import analyse_dataset, dataset_summary  # noqa: E402
from visualise.boxes.yolo_dataset import show_dataset_samples  # noqa: E402
from visualise.charts.dataset import show_dataset_charts  # noqa: E402
from visualise.common import safe_name  # noqa: E402
from visualise.metrics.dataset import format_dataset_report  # noqa: E402

# ============================================================================ SETTINGS
CONFIG_PATH = ROOT / "configs" / "image_dataset.json"   # "analyse" holds the dataset and what to show
INPUT_DIR = None           # the YOLO dataset to analyse (its folder or its data.yaml); None = the config's
SPLITS = None              # splits to analyse, e.g. ["train", "val"]; None = the config's
SHOW_PLOTS = None          # False = do not open the figures (they are still saved); None = the config's
SAVE = None                # False = write nothing to the output folder; None = the config's
SHOW_SAMPLES = None        # True = also open a few random images with their boxes drawn; None = the config's
# =====================================================================================


def main():
    parser = argparse.ArgumentParser(description="Report what a YOLO dataset holds: splits, videos and boxes.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--input", default=INPUT_DIR, help="YOLO dataset to analyse: its folder or its data.yaml.")
    parser.add_argument("--splits", nargs="+", default=SPLITS, help="Splits to analyse, e.g. train val.")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=SHOW_PLOTS,
                        help="Open the pie charts and the box plots.")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=SAVE,
                        help="Save the report and the figures to the output folder.")
    parser.add_argument("--samples", action=argparse.BooleanOptionalAction, default=SHOW_SAMPLES,
                        help="Also open a few random images with their boxes drawn.")
    args = parser.parse_args()

    if not Path(args.config).is_file():
        raise SystemExit(f"Config not found: {args.config}\n"
                         f"Copy configs/image_dataset.example.json to configs/image_dataset.json "
                         f"and fill it in.")
    config = load_config(args.config)
    if "analyse" not in config:
        raise SystemExit("config has no 'analyse' section. See configs/image_dataset.example.json.")
    overrides = {"input_dir": args.input, "splits": args.splits, "show_plots": args.show, "save": args.save}
    s = {**config["analyse"], **{k: v for k, v in overrides.items() if v is not None}}
    samples = dict(s.get("samples") or {})
    if args.samples is not None:
        samples["enabled"] = args.samples

    dataset_dir = Path(s["input_dir"]).resolve()
    if dataset_dir.is_file():   # the dataset's data.yaml was given: the dataset is its folder
        dataset_dir = dataset_dir.parent
    stats = analyse_dataset(dataset_dir, splits=s.get("splits"), prefix_separator=s.get("prefix_separator", "_frame_"))
    data_yaml = dataset_dir / "data.yaml"
    class_names = read_class_names(data_yaml if data_yaml.is_file() else None)

    report = format_dataset_report(dataset_dir.name, stats, class_names)
    print("\n" + report)

    out_dir = None
    if s.get("save", True):
        out_dir = Path(s.get("output_dir") or Path(config["output_dir"]) / "dataset_analysis" / safe_name(dataset_dir.name))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.txt").write_text(report)
        (out_dir / "summary.json").write_text(json.dumps(dataset_summary(stats, class_names), indent=2))
        print(f"Report and figures saved to: {out_dir}")
    show_dataset_charts(stats, class_names, dataset_dir.name, out_dir=out_dir, show=s.get("show_plots", True))

    if samples.get("enabled"):
        split = samples.get("split") or stats[0].name
        show_dataset_samples(dataset_dir, split=split, num_samples=int(samples.get("num_samples", 5)),
                             class_names=class_names or None)


if __name__ == "__main__":
    main()
