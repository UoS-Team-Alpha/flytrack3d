"""Convert a detection dataset between the YOLO and COCO formats.

    yolo_to_coco  a YOLO dataset folder (data.yaml, images/, labels/) -> COCO:
                  <output>/data/ and <output>/annotations/instances_<split>.json (needs fiftyone)
    coco_to_yolo  a COCO dataset folder in that layout -> YOLO:
                  <output>/images/<split>/, labels/<split>/ and data.yaml (needs ultralytics)

The input dataset is never changed, and the output folder must not exist yet
(or be empty). In Python, the two conversions are:

    from dataset_tools.image_tools.conversion.convert_yolo_to_coco import convert_yolo_to_coco
    from dataset_tools.image_tools.conversion.convert_coco_to_yolo import convert_coco_to_yolo

    coco_dir = convert_yolo_to_coco("datasets/flies", "datasets/flies_coco", splits=("train", "val"))
    data_yaml = convert_coco_to_yolo("datasets/flies_coco", "datasets/flies_yolo")

From the IDE: edit the SETTINGS block below and press Run (or F5).
From the command line, flags override the settings:
    python pipeline/step1_image_dataset_editing/main_convert_dataset_format.py --direction yolo_to_coco --input datasets/flies --output datasets/flies_coco
    python pipeline/step1_image_dataset_editing/main_convert_dataset_format.py --direction coco_to_yolo --input datasets/flies_coco --output datasets/flies_yolo --symlink
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # the repository root
sys.path.insert(0, str(ROOT))                # so the project packages import when this file is run directly

from dataset_tools.image_tools.conversion.convert_coco_to_yolo import convert_coco_to_yolo  # noqa: E402
from dataset_tools.image_tools.conversion.convert_yolo_to_coco import convert_yolo_to_coco  # noqa: E402

# ============================================================================ SETTINGS
DIRECTION = "yolo_to_coco"         # "yolo_to_coco" or "coco_to_yolo"
INPUT_DIR = "path/to/dataset"      # the dataset to convert (YOLO: the folder holding data.yaml)
OUTPUT_DIR = "path/to/dataset_coco"   # where the converted dataset is written; must not exist yet
SPLITS = ["train", "val"]          # splits to convert; None = every split found (coco_to_yolo only)
COPY_IMAGES = True                 # False = symlink the images instead of copying them
# =====================================================================================


def main():
    parser = argparse.ArgumentParser(description="Convert a detection dataset between the YOLO and COCO formats.")
    parser.add_argument("--direction", default=DIRECTION, choices=["yolo_to_coco", "coco_to_yolo"])
    parser.add_argument("--input", default=INPUT_DIR, help="Dataset folder to convert.")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Folder for the converted dataset (must not exist yet).")
    parser.add_argument("--splits", nargs="+", default=SPLITS, help="Splits to convert, e.g. train val.")
    parser.add_argument("--symlink", action="store_true", default=not COPY_IMAGES,
                        help="Symlink the images instead of copying them.")
    args = parser.parse_args()

    if args.direction == "yolo_to_coco":
        if args.splits is None:
            raise SystemExit("yolo_to_coco needs the splits to convert, e.g. --splits train val")
        coco_dir = convert_yolo_to_coco(args.input, args.output, splits=args.splits, copy_images=not args.symlink)
        print(f"\nConversion complete! COCO dataset saved to: {coco_dir}")
    else:
        data_yaml = convert_coco_to_yolo(args.input, args.output, splits=args.splits, copy_images=not args.symlink)
        print(f"\nConversion complete! YOLO dataset yaml: {data_yaml}")


if __name__ == "__main__":
    main()
