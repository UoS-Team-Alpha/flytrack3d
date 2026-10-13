"""Run detectors on new recordings: predictions only, no ground truth, no evaluation.

The run folder has the same layout as a batch evaluation run
(run_<datetime>/<input name>/<model>/predictions/), so main_points3d.py,
main_tracks3d.py and main_tracks3d_guided.py can use it directly. Name each
camera's video so it contains the camera name (see "cameras" in the config).

Inputs are the config's "dataset" (a batch's "inputs" folder, or its one video /
image folder; "labels" is ignored), or the folder given with --inputs. Batch
settings such as label_template and start_index still number the frames.
Predictions cached by earlier runs (same model, same frames) are reused.

From the IDE: edit the SETTINGS block below and press Run (or F5).
From the command line, flags override the settings:
    python pipeline/step5_inference/main_inference2d.py --inputs path/to/new_videos
    python pipeline/step5_inference/main_inference2d.py --models yolo_s_12_pruned_640_sahi --force-inference
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # the repository root
sys.path.insert(0, str(ROOT))                # so the project packages import when this file is run directly

from evaluate2d.config import load_config  # noqa: E402
from evaluate2d.evaluate_models import run_inference_only  # noqa: E402

# ============================================================================ SETTINGS
CONFIG_PATH = ROOT / "config.json"   # models; "dataset" gives the inputs unless INPUTS is set
INPUTS = None              # folder of videos / frame folders; None = the config's dataset
MODELS = None              # e.g. ["yolo_s_12_base_sahi"]; None = models with "enabled": true in config
FORCE_INFERENCE = False    # True = ignore cached predictions and re-run the detector
# =====================================================================================


def main():
    parser = argparse.ArgumentParser(description="Run object detectors without evaluating them.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--inputs", default=INPUTS,
                        help="Folder of videos / frame folders (overrides the config's dataset).")
    parser.add_argument("--models", nargs="+", default=MODELS,
                        help="Model names from config to run (overrides 'enabled').")
    parser.add_argument("--force-inference", action=argparse.BooleanOptionalAction, default=FORCE_INFERENCE,
                        help="Re-run inference even if cached predictions exist.")
    args = parser.parse_args()

    config = load_config(args.config)
    inputs = str(Path(args.inputs).resolve()) if args.inputs else None
    run_inference_only(config, force_inference=args.force_inference, only_models=args.models, inputs=inputs)


if __name__ == "__main__":
    main()
