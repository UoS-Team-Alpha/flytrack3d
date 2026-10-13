"""Convert a detection dataset between the YOLO and COCO formats.

    yolo_to_coco  a YOLO dataset folder (data.yaml, images/, labels/) -> COCO:
                  <output>/data/ and <output>/annotations/instances_<split>.json (needs fiftyone)
    coco_to_yolo  a COCO dataset folder in that layout -> YOLO:
                  <output>/images/<split>/, labels/<split>/ and data.yaml (needs ultralytics)

The direction, the dataset and the output folder come from the "convert_format"
section of the image dataset config. The input dataset is never changed, and
the output folder must not exist yet (or be empty). In Python, the two
conversions are:

    from dataset_tools.image_tools.conversion.convert_yolo_to_coco import convert_yolo_to_coco
    from dataset_tools.image_tools.conversion.convert_coco_to_yolo import convert_coco_to_yolo

    coco_dir = convert_yolo_to_coco("datasets/flies", "datasets/flies_coco", splits=("train", "val"))
    data_yaml = convert_coco_to_yolo("datasets/flies_coco", "datasets/flies_yolo")

From the IDE: edit the SETTINGS block below and press Run (or F5).
From the command line, flags override the settings and the config:
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
from evaluate2d.config import load_config  # noqa: E402

# ============================================================================ SETTINGS
CONFIG_PATH = ROOT / "image_dataset_config.json"   # "convert_format" holds the direction, dataset and output folder
DIRECTION = None           # "yolo_to_coco" or "coco_to_yolo"; None = the config's
INPUT_DIR = None           # the dataset to convert (YOLO: the folder holding data.yaml); None = the config's
OUTPUT_DIR = None          # where the converted dataset is written, must not exist yet; None = the config's
SPLITS = None              # splits to convert, e.g. ["train", "val"]; None = the config's
COPY_IMAGES = None         # False = symlink the images instead of copying them; None = the config's
# =====================================================================================


def main():
    parser = argparse.ArgumentParser(description="Convert a detection dataset between the YOLO and COCO formats.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--direction", default=DIRECTION, choices=["yolo_to_coco", "coco_to_yolo"])
    parser.add_argument("--input", default=INPUT_DIR, help="Dataset folder to convert.")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Folder for the converted dataset (must not exist yet).")
    parser.add_argument("--splits", nargs="+", default=SPLITS, help="Splits to convert, e.g. train val.")
    parser.add_argument("--symlink", action=argparse.BooleanOptionalAction,
                        default=None if COPY_IMAGES is None else not COPY_IMAGES,
                        help="Symlink the images instead of copying them.")
    args = parser.parse_args()

    if not Path(args.config).is_file():
        raise SystemExit(f"Config not found: {args.config}\n"
                         f"Copy config_examples/image_dataset_config.example.json to image_dataset_config.json "
                         f"and fill it in.")
    config = load_config(args.config)
    if "convert_format" not in config:
        raise SystemExit("config has no 'convert_format' section. See config_examples/image_dataset_config.example.json.")
    overrides = {"direction": args.direction, "input_dir": args.input, "output_dir": args.output,
                 "splits": args.splits, "copy_images": None if args.symlink is None else not args.symlink}
    s = {**config["convert_format"], **{k: v for k, v in overrides.items() if v is not None}}
    copy_images = s.get("copy_images", True)

    if s["direction"] == "yolo_to_coco":
        if not s.get("splits"):
            raise SystemExit("yolo_to_coco needs the splits to convert, e.g. \"splits\": [\"train\", \"val\"]")
        coco_dir = convert_yolo_to_coco(s["input_dir"], s["output_dir"], splits=s["splits"], copy_images=copy_images)
        print(f"\nConversion complete! COCO dataset saved to: {coco_dir}")
    elif s["direction"] == "coco_to_yolo":
        data_yaml = convert_coco_to_yolo(s["input_dir"], s["output_dir"], splits=s.get("splits"),
                                         copy_images=copy_images)
        print(f"\nConversion complete! YOLO dataset yaml: {data_yaml}")
    else:
        raise SystemExit(f"convert_format.direction must be 'yolo_to_coco' or 'coco_to_yolo', got '{s['direction']}'")


if __name__ == "__main__":
    main()
