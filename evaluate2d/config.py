"""Config loading shared by every main_*.py: refuse templates, make paths absolute."""

from __future__ import annotations

import json
from pathlib import Path

PATH_KEYS = {"images", "labels", "video", "inputs", "data_yaml", "model_path", "config_path", "output_dir"}


def resolve_paths(obj, base: Path):
    """Make relative paths in the config relative to the config file's folder."""
    if isinstance(obj, dict):
        return {k: (str((base / v).resolve()) if k in PATH_KEYS and isinstance(v, str) and v
                    else resolve_paths(v, base)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_paths(v, base) for v in obj]
    return obj


def load_config(path: str | Path) -> dict:
    """Read a config, refuse templates and make its paths absolute. Shared by every main_*.py."""
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}\n"
                         f"Copy config_examples/config.example.json to config.json and fill in your paths.")
    raw = json.loads(config_path.read_text())
    if raw.get("_example"):
        raise SystemExit(f"{config_path.name} is a template and cannot be run.\n"
                         f"Copy it to config.json, fill in your paths and remove the '_example' line.")
    config = resolve_paths(raw, config_path.parent)
    config.setdefault("output_dir", str(config_path.parent / "output"))
    return config
