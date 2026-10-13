"""Config loading shared by every main_*.py: refuse templates, make paths absolute."""

from __future__ import annotations

import json
from pathlib import Path

PATH_KEYS = {"images", "labels", "video", "inputs", "videos_dir", "data_yaml", "model_path", "config_path", "output_dir",
             "p2_model_yaml", "weights_dir"}


def resolve_paths(obj, base: Path):
    """Make relative paths in the config relative to the config file's folder."""
    if isinstance(obj, dict):
        return {k: (str((base / v).resolve()) if k in PATH_KEYS and isinstance(v, str) and v
                    else resolve_paths(v, base)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_paths(v, base) for v in obj]
    return obj


def resolve_trained_runs(raw: dict, base: Path) -> None:
    """A model_path that is a bare name (no folder, no extension) names a run of main_train_detector.py:
    it becomes that run's best.pt, <output_dir>/training/<training.wandb_project>/<name>/weights/best.pt."""
    from training.yolo import trained_weights

    for model in raw.get("models", []):
        run = model.get("model_path")
        if not isinstance(run, str) or not run or "/" in run or "\\" in run or Path(run).suffix:
            continue
        project = raw.get("training", {}).get("wandb_project")
        if not project:
            raise SystemExit(f"Model '{model.get('name')}': model_path '{run}' is a training run name, "
                             f"but the config has no training.wandb_project to look it up in.")
        weights = trained_weights((base / raw.get("output_dir", "output")).resolve(), project, run)
        if model.get("enabled") and not weights.is_file():
            runs = sorted(d.name for d in weights.parents[2].glob("*") if (d / "weights" / "best.pt").is_file())
            raise SystemExit(f"Model '{model.get('name')}': no trained run '{run}' ({weights} not found).\n"
                             f"Runs in {weights.parents[2]}: {', '.join(runs) or 'none'}")
        model["model_path"] = str(weights)


def load_config(path: str | Path) -> dict:
    """Read a config, refuse templates and make its paths absolute. Shared by every main_*.py."""
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}\n"
                         f"Copy config_examples/config.example.json to config.json and fill in your paths.")
    raw = json.loads(config_path.read_text())
    if raw.get("_example"):
        raise SystemExit(f"{config_path.name} is a template and cannot be run.\n"
                         f"Copy it to the project root (e.g. as {config_path.name.replace('.example', '')}), "
                         f"fill in your paths and remove the '_example' line.")
    resolve_trained_runs(raw, config_path.parent)
    config = resolve_paths(raw, config_path.parent)
    config.setdefault("output_dir", str(config_path.parent / "output"))
    return config
