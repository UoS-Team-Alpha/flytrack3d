"""Merge a folder of YOLO datasets into one dataset, without ever adding the same image twice.

Every sub-folder of the input folder that holds a split (images/<split> + labels/<split>)
is one dataset to merge. Its labelled images are copied to the output as

    <output>/images/<split>/<dataset>_<image>.<ext>, <output>/labels/<split>/<dataset>_<image>.txt, <split>.txt and data.yaml

The output may already hold a dataset, which is then added to. An image is only added when
no image with the same content is in the output yet (in any split): the images already
there and every image to add are hashed (the whole file) and compared. So a frame that is
in two of the datasets, or a dataset that is merged a second time, ends up in the output
once. The duplicates that were skipped are reported, as are those whose labels differ from
the copy that was kept. Nothing in the output is overwritten and the datasets to merge are
never changed.

The folders come from the "merge" section of the image dataset config.

From the IDE: edit the SETTINGS block below and press Run (or F5).
From the command line, flags override the settings and the config:
    python pipeline/step1_image_dataset_creation/main_merge_datasets.py
    python pipeline/step1_image_dataset_creation/main_merge_datasets.py --input datasets/sessions --output datasets/merged
    python pipeline/step1_image_dataset_creation/main_merge_datasets.py --splits train --keep-classes 0 --no-keep-empty
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # the repository root
sys.path.insert(0, str(ROOT))                # so the project packages import when this file is run directly

from common.config import load_config  # noqa: E402
from dataset_tools.image_tools.merging.merge_datasets import merge_datasets  # noqa: E402

# ============================================================================ SETTINGS
CONFIG_PATH = ROOT / "image_dataset_config.json"   # "merge" holds the folders and what to keep
INPUT_DIR = None           # the folder holding the datasets to merge, one per sub-folder; None = the config's
OUTPUT_DIR = None          # the merged dataset; may exist already, it is then added to; None = the config's
SPLITS = None              # the splits to merge, e.g. ["train"]; None = the config's
KEEP_CLASSES = None        # class ids to keep, renumbered from 0, e.g. [0]; None = the config's
CLASS_NAMES = None         # names of the merged dataset's classes, e.g. ["fly"]; None = the config's
KEEP_EMPTY = None          # True = also add images without a (kept) box, as background images; None = the config's
# =====================================================================================


def main():
    parser = argparse.ArgumentParser(description="Merge a folder of YOLO datasets into one, without duplicate images.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--input", default=INPUT_DIR, help="Folder holding the datasets to merge, one per sub-folder.")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Folder of the merged dataset (added to if it exists).")
    parser.add_argument("--splits", nargs="+", default=SPLITS, help="Splits to merge; default: every one found.")
    parser.add_argument("--keep-classes", nargs="+", type=int, default=KEEP_CLASSES,
                        help="Class ids to keep, renumbered from 0 in this order; default: every class.")
    parser.add_argument("--class-names", nargs="+", default=CLASS_NAMES,
                        help="Names of the merged dataset's classes; default: read from the datasets.")
    parser.add_argument("--keep-empty", action=argparse.BooleanOptionalAction, default=KEEP_EMPTY,
                        help="Also add images without a (kept) box, as background images.")
    args = parser.parse_args()

    if not Path(args.config).is_file():
        raise SystemExit(f"Config not found: {args.config}\n"
                         f"Copy config_examples/image_dataset_config.example.json to image_dataset_config.json "
                         f"and fill it in.")
    config = load_config(args.config)
    if "merge" not in config:
        raise SystemExit("config has no 'merge' section. See config_examples/image_dataset_config.example.json.")
    overrides = {"input_dir": args.input, "output_dir": args.output, "splits": args.splits,
                 "keep_classes": args.keep_classes, "class_names": args.class_names, "keep_empty": args.keep_empty}
    s = {**config["merge"], **{k: v for k, v in overrides.items() if v is not None}}

    data_yaml = merge_datasets(s["input_dir"], s["output_dir"], splits=s.get("splits"),
                               keep_classes=s.get("keep_classes"), class_names=s.get("class_names"),
                               keep_empty=bool(s.get("keep_empty", True)))
    print(f"Dataset yaml: {data_yaml}")


if __name__ == "__main__":
    main()
