"""Run detector evaluation.

From the IDE: edit the SETTINGS block below and press Run (or F5).
From the command line, flags override the settings:
    python main.py --config other.json
    python main.py --models yolo_s_12_base_sahi yolo_s_11_p2_sahi
    python main.py --force-inference --video --no-track-plot
    python main.py --config config.batch.json    # folder of videos + folder of labels

Dataset types: a folder of images ("yolo_images"), one video ("video"), or a
folder of videos / frame folders paired with a folder of label folders
("batch"). See config.example.json and config.batch.example.json.
"""

import argparse
import json
from pathlib import Path

from evaluate_models import run_evaluation

HERE = Path(__file__).resolve().parent

# ============================================================================ SETTINGS
CONFIG_PATH = HERE / "config.json"   # dataset, models, metric + visualisation settings
MODELS = None              # e.g. ["yolo_s_12_base_sahi"]; None = models with "enabled": true in config
FORCE_INFERENCE = False    # True = ignore cached predictions and re-run the detector
SAVE_VIDEO = False         # annotated MP4 per sequence (GT green, predictions magenta)
SAVE_TRACK_PLOT = True     # frame-sized PNG per sequence with GT vs predicted tracks
# =====================================================================================

PATH_KEYS = {"images", "labels", "video", "inputs", "data_yaml", "model_path", "config_path", "output_dir"}


def resolve_paths(obj, base: Path):
    """Make relative paths in the config relative to the config file's folder."""
    if isinstance(obj, dict):
        return {k: (str((base / v).resolve()) if k in PATH_KEYS and isinstance(v, str) and v
                    else resolve_paths(v, base)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_paths(v, base) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(description="Evaluate object detectors (COCO-style metrics).")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--models", nargs="+", default=MODELS,
                        help="Model names from config to run (overrides 'enabled').")
    parser.add_argument("--force-inference", action=argparse.BooleanOptionalAction, default=FORCE_INFERENCE,
                        help="Re-run inference even if cached predictions exist.")
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=SAVE_VIDEO,
                        help="Save annotated videos.")
    parser.add_argument("--track-plot", action=argparse.BooleanOptionalAction, default=SAVE_TRACK_PLOT,
                        help="Save GT vs prediction track plots.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}\n"
                         f"Copy config.example.json to config.json and fill in your paths.")
    raw = json.loads(config_path.read_text())
    if raw.get("_example"):
        raise SystemExit(f"{config_path.name} is a template and cannot be run.\n"
                         f"Copy it to config.json, fill in your paths and remove the '_example' line.")
    config = resolve_paths(raw, config_path.parent)
    config.setdefault("output_dir", str(config_path.parent / "output"))
    run_evaluation(config, force_inference=args.force_inference, only_models=args.models,
                   save_video=args.video, save_track_plot=args.track_plot)


if __name__ == "__main__":
    main()
