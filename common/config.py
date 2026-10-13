"""Config loading shared by every main_*.py: merge includes, refuse templates, make paths absolute."""

from __future__ import annotations

import json
import re
from pathlib import Path

PATH_KEYS = {"images", "labels", "video", "inputs", "videos_dir", "annotations_zip", "input_dir", "data_yaml",
             "model_path", "config_path", "output_dir", "p2_model_yaml", "weights_dir", "csv", "fg_labels", "videos", "save"}
ROOT = Path(__file__).resolve().parents[1]   # the repository root


def is_run_name(value) -> bool:
    """A model_path that is a bare name (no folder, no extension) names a run of main_train_detector.py."""
    return isinstance(value, str) and bool(value) and "/" not in value and "\\" not in value and not Path(value).suffix


def resolve_paths(obj, base: Path):
    """Make relative paths in the config relative to the config file's folder."""
    if isinstance(obj, dict):
        return {k: (str((base / v).resolve()) if k in PATH_KEYS and isinstance(v, str) and v
                    and not v.startswith("${") and not (k == "model_path" and is_run_name(v))
                    else resolve_paths(v, base)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_paths(v, base) for v in obj]
    return obj


def trained_weights(output_dir: str | Path, wandb_project: str, run: str) -> Path:
    """best.pt of a training run: <output_dir>/training/<wandb_project>/<run>/weights/best.pt."""
    return Path(output_dir) / "training" / wandb_project / run / "weights" / "best.pt"


def resolve_trained_runs(config: dict) -> None:
    """A model_path that is a run name becomes that run's best.pt,
    <output_dir>/training/<training.wandb_project>/<name>/weights/best.pt."""
    for model in config.get("models", []):
        run = model.get("model_path")
        if not is_run_name(run):
            continue
        project = config.get("training", {}).get("wandb_project")
        if not project:
            raise SystemExit(f"Model '{model.get('name')}': model_path '{run}' is a training run name, "
                             f"but the config has no training.wandb_project to look it up in "
                             f"(include training.json).")
        weights = trained_weights(config["output_dir"], project, run)
        if model.get("enabled") and not weights.is_file():
            runs = sorted(d.name for d in weights.parents[2].glob("*") if (d / "weights" / "best.pt").is_file())
            raise SystemExit(f"Model '{model.get('name')}': no trained run '{run}' ({weights} not found).\n"
                             f"Runs in {weights.parents[2]}: {', '.join(runs) or 'none'}")
        model["model_path"] = str(weights)


def expand_paths(config: dict) -> dict:
    """Replace every "${name}" in the config's strings with config["paths"][name], so that a path
    used by several configs is set once (configs/shared/paths.json). A path may use another one."""
    paths = {k: v for k, v in (config.get("paths") or {}).items() if isinstance(v, str)}

    def expand(text: str, seen: tuple = ()) -> str:
        def lookup(m):
            name = m.group(1)
            if name in seen:
                raise SystemExit("Paths refer to each other: " + " -> ".join((*seen, name)))
            if name not in paths:
                raise SystemExit(f"The config uses ${{{name}}}, which is not in its 'paths' "
                                 f"(known: {', '.join(paths) or 'none'}). Add it to configs/shared/paths.json, "
                                 f"and check the config includes that file.")
            return expand(paths[name], (*seen, name))
        return re.sub(r"\$\{(\w+)\}", lookup, text)

    def walk(obj):
        if isinstance(obj, dict):
            return {k: walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [walk(v) for v in obj]
        return expand(obj) if isinstance(obj, str) else obj
    return walk(config)


def merge(base: dict, over: dict) -> dict:
    """base with over's keys on top: dicts are merged key by key, anything else is replaced."""
    out = dict(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def read_config(config_path: Path, chain: tuple = ()) -> dict:
    """One config file with its paths made absolute and its "_include" files merged underneath it.

    "_include" is a file name or a list of them, relative to this file's folder. They are merged in
    order (later ones win) and this file's own keys win over all of them. An included file's
    top-level "_..." keys (its _readme) are dropped."""
    if config_path in chain:
        raise SystemExit("Configs include each other: " + " -> ".join(p.name for p in (*chain, config_path)))
    if not config_path.is_file():
        example = config_path.with_name(config_path.stem + ".example.json")
        hint = (f"Copy {example.name} to {config_path.name} (same folder) and fill it in." if example.is_file()
                else "Each config in configs/ has a <name>.example.json template next to it.")
        included = f" (included by {chain[-1]})" if chain else ""
        raise SystemExit(f"Config not found: {config_path}{included}\n{hint}")
    raw = json.loads(config_path.read_text())
    if raw.get("_example"):
        raise SystemExit(f"{config_path} is a template and cannot be run.\n"
                         f"Copy it to {config_path.name.replace('.example', '')} in the same folder, "
                         f"fill in your paths and remove the '_example' line.")
    includes = raw.pop("_include", None) or []
    config = {}
    for name in [includes] if isinstance(includes, str) else includes:
        included = read_config((config_path.parent / name).resolve(), (*chain, config_path))
        config = merge(config, {k: v for k, v in included.items() if not k.startswith("_")})
    if isinstance(raw.get("paths"), dict):   # every entry of "paths" is a path, whatever its name
        raw["paths"] = {k: (str((config_path.parent / v).resolve()) if isinstance(v, str) and v
                            and not k.startswith("_") and not v.startswith("${") else v)
                        for k, v in raw["paths"].items()}
    return merge(config, resolve_paths(raw, config_path.parent))


def load_config(path: str | Path) -> dict:
    """Read a config and the files it includes, refuse templates, fill in the "${name}" paths
    and make its paths absolute.
    Shared by every main_*.py."""
    config = expand_paths(read_config(Path(path).resolve()))
    config.setdefault("output_dir", str(ROOT / "output"))
    resolve_trained_runs(config)
    return config
