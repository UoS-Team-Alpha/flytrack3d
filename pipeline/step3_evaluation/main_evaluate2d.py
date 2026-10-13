"""Run detector evaluation.

From the IDE: edit the SETTINGS block below and press Run (or F5).
From the command line, flags override the settings:
    python pipeline/step3_evaluation/main_evaluate2d.py --config other.json
    python pipeline/step3_evaluation/main_evaluate2d.py --models yolo_s_12_base_sahi yolo_s_11_p2_sahi
    python pipeline/step3_evaluation/main_evaluate2d.py --force-inference --video --no-track-plot
    python pipeline/step3_evaluation/main_evaluate2d.py --config config.batch.json    # folder of videos + folder of labels

Dataset types: a folder of images ("yolo_images"), one video ("video"), or a
folder of videos / frame folders paired with a folder of label folders
("batch"). See config_examples/config.example.json and config_examples/config.batch.example.json.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # the repository root
sys.path.insert(0, str(ROOT))                # so the project packages import when this file is run directly

from evaluate2d.config import load_config  # noqa: E402
from evaluate2d.evaluate_models import run_evaluation  # noqa: E402

# ============================================================================ SETTINGS
CONFIG_PATH = ROOT / "config.json"   # dataset, models, metric + visualisation settings
MODELS = None              # e.g. ["yolo_s_12_base_sahi"]; None = models with "enabled": true in config
FORCE_INFERENCE = False    # True = ignore cached predictions and re-run the detector
SAVE_VIDEO = False         # annotated MP4 per sequence (GT green, predictions magenta)
SAVE_TRACK_PLOT = True     # frame-sized PNG per sequence with GT vs predicted tracks
# =====================================================================================


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

    config = load_config(args.config)
    run_evaluation(config, force_inference=args.force_inference, only_models=args.models,
                   save_video=args.video, save_track_plot=args.track_plot)


if __name__ == "__main__":
    main()
