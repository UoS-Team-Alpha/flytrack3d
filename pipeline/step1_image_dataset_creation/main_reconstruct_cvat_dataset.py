"""Rebuild a YOLO dataset from a CVAT annotation export and the video it was annotated on.

A CVAT export of a video task holds the labels but not the images. This takes
the export zip and the video, pulls the labelled frames out of the video and
writes them with their labels as a YOLO dataset:

    <output>/images/train/frame_<n>.png, <output>/labels/train/frame_<n>.txt, train.txt and data.yaml

The export zip, the video and the output folder come from the "reconstruct_cvat"
section of the image dataset config. Only frames with annotations are kept
(--keep-empty also keeps the empty ones, as background images). The class names
are read from the export. The output folder must not exist yet (or be empty).
Needs opencv.

From the IDE: edit the SETTINGS block below and press Run (or F5).
From the command line, flags override the settings and the config:
    python pipeline/step1_image_dataset_creation/main_reconstruct_cvat_dataset.py
    python pipeline/step1_image_dataset_creation/main_reconstruct_cvat_dataset.py --annotations export.zip --video session1.mp4 --output datasets/session1
    python pipeline/step1_image_dataset_creation/main_reconstruct_cvat_dataset.py --keep-empty --classes fly
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # the repository root
sys.path.insert(0, str(ROOT))                # so the project packages import when this file is run directly

from dataset_tools.image_tools.cvat.reconstruct_dataset import reconstruct_cvat_dataset  # noqa: E402
from evaluate2d.config import load_config  # noqa: E402

# ============================================================================ SETTINGS
CONFIG_PATH = ROOT / "image_dataset_config.json"   # "reconstruct_cvat" holds the zip, video and output folder
ANNOTATIONS_ZIP = None     # the CVAT export (YOLO format) of the video task; None = the config's
VIDEO = None               # the video that was annotated; None = the config's
OUTPUT_DIR = None          # where the dataset is written, must not exist yet; None = the config's
CLASS_NAMES = None         # classes in label order, e.g. ["fly"]; None = the config's
KEEP_EMPTY = None          # True = also keep frames with an empty label file; None = the config's
# =====================================================================================


def main():
    parser = argparse.ArgumentParser(description="Rebuild a YOLO dataset from a CVAT export and its video.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--annotations", default=ANNOTATIONS_ZIP, help="CVAT annotation export (.zip).")
    parser.add_argument("--video", default=VIDEO, help="The video that was annotated.")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Folder for the dataset (must not exist yet).")
    parser.add_argument("--classes", nargs="+", default=CLASS_NAMES,
                        help="Class names in label order; default: read from the export.")
    parser.add_argument("--keep-empty", action=argparse.BooleanOptionalAction, default=KEEP_EMPTY,
                        help="Also keep frames with an empty label file, as background images.")
    args = parser.parse_args()

    if not Path(args.config).is_file():
        raise SystemExit(f"Config not found: {args.config}\n"
                         f"Copy config_examples/image_dataset_config.example.json to image_dataset_config.json "
                         f"and fill it in.")
    config = load_config(args.config)
    if "reconstruct_cvat" not in config:
        raise SystemExit("config has no 'reconstruct_cvat' section. See config_examples/image_dataset_config.example.json.")
    overrides = {"annotations_zip": args.annotations, "video": args.video, "output_dir": args.output,
                 "class_names": args.classes, "keep_empty": args.keep_empty}
    s = {**config["reconstruct_cvat"], **{k: v for k, v in overrides.items() if v is not None}}

    data_yaml = reconstruct_cvat_dataset(s["annotations_zip"], s["video"], s["output_dir"],
                                         class_names=s.get("class_names"), keep_empty=bool(s.get("keep_empty")))
    print(f"Dataset yaml: {data_yaml}")


if __name__ == "__main__":
    main()
