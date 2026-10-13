"""Show a few random images of a YOLO dataset with their bounding boxes drawn, to check the labels line up.

The images are opened one at a time, each with its boxes and their class names drawn on it
(press any key for the next one), as after tiling. The dataset comes from the "check_bboxes"
section of the image dataset config and is never changed; nothing is saved.

From the IDE: edit the SETTINGS block below and press Run (or F5).
From the command line, flags override the settings and the config:
    python pipeline/step1_image_dataset_creation/main_check_bboxes.py
    python pipeline/step1_image_dataset_creation/main_check_bboxes.py -n 20
    python pipeline/step1_image_dataset_creation/main_check_bboxes.py --input datasets/flies --split val --seed 0
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # the repository root
sys.path.insert(0, str(ROOT))                # so the project packages import when this file is run directly

from common.config import load_config  # noqa: E402
from common.datasets import read_class_names  # noqa: E402
from common.yolo_format import find_splits  # noqa: E402
from visualise.boxes.yolo_dataset import show_dataset_samples  # noqa: E402

# ============================================================================ SETTINGS
CONFIG_PATH = ROOT / "image_dataset_config.json"   # "check_bboxes" holds the dataset and what to show
INPUT_DIR = None           # the YOLO dataset to check (its folder or its data.yaml); None = the config's
SPLIT = None               # the split to take the images from, e.g. "val"; None = the config's
NUM_SAMPLES = None         # how many images to show; None = the config's
SEED = None                # a number = the same images every run; None = the config's
# =====================================================================================


def main():
    parser = argparse.ArgumentParser(description="Show random images of a YOLO dataset with their boxes drawn.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--input", default=INPUT_DIR, help="YOLO dataset to check: its folder or its data.yaml.")
    parser.add_argument("--split", default=SPLIT, help="Split to take the images from, e.g. val.")
    parser.add_argument("-n", "--num-samples", type=int, default=NUM_SAMPLES, help="How many images to show.")
    parser.add_argument("--seed", type=int, default=SEED, help="Show the same images every run.")
    args = parser.parse_args()

    if not Path(args.config).is_file():
        raise SystemExit(f"Config not found: {args.config}\n"
                         f"Copy config_examples/image_dataset_config.example.json to image_dataset_config.json "
                         f"and fill it in.")
    config = load_config(args.config)
    if "check_bboxes" not in config:
        raise SystemExit("config has no 'check_bboxes' section. See config_examples/image_dataset_config.example.json.")
    overrides = {"input_dir": args.input, "split": args.split, "num_samples": args.num_samples, "seed": args.seed}
    s = {**config["check_bboxes"], **{k: v for k, v in overrides.items() if v is not None}}

    dataset_dir = Path(s["input_dir"]).resolve()
    if dataset_dir.is_file():   # the dataset's data.yaml was given: the dataset is its folder
        dataset_dir = dataset_dir.parent
    splits = find_splits(dataset_dir)
    if not splits:
        raise SystemExit(f"No split found in {dataset_dir} (images/<split>/ or <split>/images/).")
    split = s.get("split") or splits[0]
    if split not in splits:
        raise SystemExit(f"{dataset_dir} has no '{split}' split. Its splits: {', '.join(splits)}.")
    data_yaml = dataset_dir / "data.yaml"
    class_names = read_class_names(data_yaml if data_yaml.is_file() else None)

    show_dataset_samples(dataset_dir, split=split, num_samples=int(s.get("num_samples", 5)),
                         class_names=class_names or None, seed=s.get("seed"))


if __name__ == "__main__":
    main()
