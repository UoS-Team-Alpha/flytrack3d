"""Tile a full-frame YOLO dataset into small square crops, to train a detector for sliced (SAHI) inference.

Each image gives tiles cropped around its objects (every object ends up in a
tile; boxes cut to a sliver by a tile's edge are dropped) plus, for a fraction
of the images, an empty background tile. The empty tiles are spread evenly over
a 3x3 grid of the frame. See dataset_tools/image_tools/tiling/tile_dataset.py.

The dataset, the output folder and the tiling settings come from the "tile"
section of the image tiling config. The output is a YOLO dataset
(images/<split>/, labels/<split>/, data.yaml) and the input is never changed.
Afterwards, a few random tiles are shown with their boxes drawn, to check the
labels line up (press any key for the next one).

From the IDE: edit the SETTINGS block below and press Run (or F5).
From the command line, flags override the settings and the config:
    python pipeline/step2_image_tiling/main_tile_dataset.py
    python pipeline/step2_image_tiling/main_tile_dataset.py --fraction 0.1 --overwrite    # quick test on 10% of each split
    python pipeline/step2_image_tiling/main_tile_dataset.py --input datasets/full --output datasets/tiles --no-visualise
    python pipeline/step2_image_tiling/main_tile_dataset.py --visualise-only              # view tiles made earlier
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # the repository root
sys.path.insert(0, str(ROOT))                # so the project packages import when this file is run directly

from dataset_tools.image_tools.tiling.tile_dataset import TileSettings, tile_dataset  # noqa: E402
from common.config import load_config  # noqa: E402
from visualise.boxes.yolo_dataset import show_dataset_samples  # noqa: E402

# ============================================================================ SETTINGS
CONFIG_PATH = ROOT / "image_tiling_config.json"   # "tile" holds the dataset, output folder and tiling settings
INPUT_DIR = None           # the full-frame YOLO dataset to tile; None = the config's
OUTPUT_DIR = None          # where the tiled dataset is written; None = the config's
SPLITS = None              # splits to tile, e.g. ["train", "val"]; None = the config's
DATASET_FRACTION = None    # fraction of each split to tile, e.g. 0.1 for a quick test; None = the config's
OVERWRITE = None           # True = delete an existing output folder and tile again; None = the config's
VISUALISE = None           # True = show sample tiles afterwards; None = the config's
VISUALISE_ONLY = False     # True = skip the tiling and only show sample tiles of the output folder
# =====================================================================================


def main():
    parser = argparse.ArgumentParser(description="Tile a full-frame YOLO dataset into small square crops.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--input", default=INPUT_DIR, help="Full-frame YOLO dataset to tile.")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Folder for the tiled dataset.")
    parser.add_argument("--splits", nargs="+", default=SPLITS, help="Splits to tile, e.g. train val.")
    parser.add_argument("--fraction", type=float, default=DATASET_FRACTION,
                        help="Fraction of each split to tile, e.g. 0.1 for a quick test.")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=OVERWRITE,
                        help="Delete an existing output folder and tile again.")
    parser.add_argument("--visualise", action=argparse.BooleanOptionalAction, default=VISUALISE,
                        help="Show sample tiles with their boxes afterwards.")
    parser.add_argument("--visualise-only", action="store_true", default=VISUALISE_ONLY,
                        help="Skip the tiling and only show sample tiles of the output folder.")
    args = parser.parse_args()

    if not Path(args.config).is_file():
        raise SystemExit(f"Config not found: {args.config}\n"
                         f"Copy config_examples/image_tiling_config.example.json to image_tiling_config.json "
                         f"and fill it in.")
    config = load_config(args.config)
    if "tile" not in config:
        raise SystemExit("config has no 'tile' section. See config_examples/image_tiling_config.example.json.")
    overrides = {"input_dir": args.input, "output_dir": args.output, "splits": args.splits,
                 "dataset_fraction": args.fraction, "overwrite": args.overwrite}
    t = {**config["tile"], **{k: v for k, v in overrides.items() if v is not None}}
    vis = dict(t.get("visualise") or {})
    if args.visualise is not None:
        vis["enabled"] = args.visualise

    if not args.visualise_only:
        settings = TileSettings(crop_size=int(t["crop_size"]), empty_fraction=float(t["empty_fraction"]),
                                empty_tries=int(t["empty_tries"]),
                                min_visible_fraction=float(t["min_visible_fraction"]),
                                min_visible_area=float(t["min_visible_area"]))
        output_dir = tile_dataset(t["input_dir"], t["output_dir"], splits=t["splits"], settings=settings,
                                  dataset_fraction=float(t.get("dataset_fraction", 1.0)), seed=t.get("seed"),
                                  overwrite=bool(t.get("overwrite")))
        print(f"\nTiling complete! Tiled dataset saved to: {output_dir}")

    if args.visualise_only or vis.get("enabled"):
        show_dataset_samples(t["output_dir"], split=vis.get("split", "train"), num_samples=int(vis.get("num_samples", 5)))


if __name__ == "__main__":
    main()
