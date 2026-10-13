"""Train a YOLO detector from the config's "training" section.

The dataset (data_yaml), GPU, W&B project / group, model (YOLO version, size,
P2 head, pretrained weights) and the Ultralytics training arguments
("train_args") all come from the config. The run is written to
output/training/<wandb_project>/<run name>/ (e.g. yolo_s_12_base; _2, _3, ...
if that name is taken), with the trained weights in its weights/ folder. To
evaluate it, give the run name as a model's "model_path" in the config.

From the IDE: edit the SETTINGS block below and press Run (or F5).
From the command line, flags override the settings and the config:
    python pipeline/step2_training/main_train_detector.py --gpu 1 --version 11 --p2
    python pipeline/step2_training/main_train_detector.py --version 12 --size s --tag pruned
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # the repository root
sys.path.insert(0, str(ROOT))                # so the project packages import when this file is run directly

from evaluate2d.config import load_config  # noqa: E402
from training.yolo import train_detector  # noqa: E402

# ============================================================================ SETTINGS
CONFIG_PATH = ROOT / "config.json"   # "training" holds the dataset, model and training arguments
GPU = None                 # GPU id(s) to train on, e.g. 0 or [0, 1]; None = the config's
VERSION = None             # YOLO version, e.g. 11 or 12; None = the config's
SIZE = None                # model size: "n", "s", "m", "l" or "x"; None = the config's
P2 = None                  # True = P2 head (needs p2_model_yaml); None = the config's
LOAD_PRETRAINED = None     # True = start from the standard model's weights; None = the config's
TAG = None                 # suffix for the run name, e.g. "pruned"; None = the config's
# =====================================================================================


def main():
    parser = argparse.ArgumentParser(description="Train a YOLO detector.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--gpu", type=int, nargs="+", default=GPU, help="GPU id(s) to train on.")
    parser.add_argument("--version", type=int, default=VERSION, help="YOLO version, e.g. 11 or 12.")
    parser.add_argument("--size", default=SIZE, choices=["n", "s", "m", "l", "x"], help="Model size.")
    parser.add_argument("--p2", action=argparse.BooleanOptionalAction, default=P2,
                        help="Use the P2 head architecture (p2_model_yaml).")
    parser.add_argument("--load-pretrained", action=argparse.BooleanOptionalAction, default=LOAD_PRETRAINED,
                        help="Load the standard model's pretrained weights.")
    parser.add_argument("--tag", default=TAG, help="Suffix appended to the run name, e.g. pruned.")
    args = parser.parse_args()

    gpu = args.gpu[0] if isinstance(args.gpu, list) and len(args.gpu) == 1 else args.gpu
    config = load_config(args.config)
    train_detector(config, overrides={"gpu": gpu, "version": args.version, "size": args.size, "p2": args.p2,
                                      "load_pretrained": args.load_pretrained, "tag": args.tag})


if __name__ == "__main__":
    main()
