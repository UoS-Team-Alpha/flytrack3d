"""Train a YOLO detector with Ultralytics, from the config's "training" section.

The model is YOLO<version><size> (e.g. 12 + "s" = yolo12s), optionally with the
P2 head (an extra stride-4 detection layer for small objects) from the
architecture yaml in "p2_model_yaml". With load_pretrained, the standard
model's weights are loaded into it: Ultralytics matches the backbone and the
P3-P5 heads, and the new P2 layers and the 4-input Detect head start random.

Runs log to Weights & Biases (project / group from the config) when wandb is
enabled in Ultralytics ("yolo settings wandb=True").
"""

from __future__ import annotations

import os
from pathlib import Path


def weights_name(version: int, size: str, suffix: str = ".pt") -> str:
    """Ultralytics file name for a model: yolo11s.pt, yolo12s.pt, yolov8s.pt (v10 and below have a 'v')."""
    return f"yolo{'v' if version <= 10 else ''}{version}{size}{suffix}"


def run_name(t: dict) -> str:
    """W&B run name, e.g. yolo_s_12_base, yolo_s_11_p2_pruned."""
    name = f"yolo_{t['size']}_{t['version']}_{'p2' if t['p2'] else 'base'}"
    if not t["load_pretrained"]:
        name += "_scratch"
    return name + (f"_{t['tag']}" if t.get("tag") else "")


def free_run_name(project_dir: Path, name: str) -> str:
    """name, or name_2, name_3, ... if a run of that name already exists (runs are never overwritten)."""
    candidate, n = name, 2
    while (project_dir / candidate).exists():
        candidate, n = f"{name}_{n}", n + 1
    return candidate


def build_model(t: dict, weights_dir: Path):
    """The YOLO model to train: standard or P2 architecture, pretrained or from scratch."""
    from ultralytics import YOLO   # imported here, after CUDA_VISIBLE_DEVICES is set

    pretrained = weights_dir / weights_name(t["version"], t["size"])   # downloaded here if missing
    if not t["p2"]:
        return YOLO(str(pretrained)) if t["load_pretrained"] else YOLO(weights_name(t["version"], t["size"], ".yaml"))

    p2_yaml = Path(t.get("p2_model_yaml") or "")
    if not p2_yaml.is_file():
        raise SystemExit(f"training.p2 is on but p2_model_yaml was not found: {p2_yaml}")
    # Ultralytics takes the model scale from the yaml's file name, so it must match the size.
    if f"{t['version']}{t['size']}" not in p2_yaml.stem:
        raise SystemExit(f"p2_model_yaml {p2_yaml.name} does not look like YOLO{t['version']}{t['size']} "
                         f"(expected a name like {weights_name(t['version'], t['size'], '-p2.yaml')}).")
    model = YOLO(str(p2_yaml))
    return model.load(str(pretrained)) if t["load_pretrained"] else model


def train_detector(config: dict, overrides: dict | None = None) -> Path:
    """Train the model described by config["training"] (keys in overrides win). Returns the run folder."""
    if "training" not in config:
        raise SystemExit("config has no 'training' section. See config_examples/config.example.json.")
    t = {**config["training"], **{k: v for k, v in (overrides or {}).items() if v is not None}}
    data_yaml = Path(t.get("data_yaml") or "")
    if not data_yaml.is_file():
        raise SystemExit(f"training.data_yaml not found: {data_yaml}")

    # Set the CUDA devices before torch is loaded; the process then sees them as 0, 1, ...
    gpus = t["gpu"] if isinstance(t["gpu"], list) else [t["gpu"]]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    device = list(range(len(gpus))) if len(gpus) > 1 else 0

    training_dir = Path(config["output_dir"]) / "training"
    name = free_run_name(training_dir / t["wandb_project"], run_name(t))
    save_dir = training_dir / t["wandb_project"] / name
    os.environ["WANDB_PROJECT"] = t["wandb_project"]
    os.environ["WANDB_RUN_GROUP"] = t["wandb_group"]
    os.environ["WANDB_NAME"] = name

    weights_dir = Path(t.get("weights_dir") or training_dir / "pretrained")
    weights_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(t, weights_dir)

    print("==================================================")
    print(f" Launching YOLO{t['version']}{t['size']} with P2 {'on' if t['p2'] else 'off'} "
          f"({'pretrained' if t['load_pretrained'] else 'from scratch'}) on CUDA device(s) {gpus}")
    print(f" Data:   {data_yaml}")
    print(f" W&B:    {t['wandb_project']} / {t['wandb_group']} / {name}")
    print(f" Output: {save_dir}")
    print("==================================================")

    model.train(
        data=str(data_yaml),
        device=device,
        project=t["wandb_project"],   # Ultralytics names the W&B project after this
        name=name,
        save_dir=str(save_dir),       # ... while the run itself is written here
        **t.get("train_args", {}),
    )
    print(f"\nWeights: {save_dir / 'weights' / 'best.pt'}\n"
          f'To evaluate it, add a model to the config with "model_path": "{name}"')
    return save_dir
