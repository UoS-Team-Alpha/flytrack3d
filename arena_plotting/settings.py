"""Settings and caching shared by the arena objects (config["arena_plotting"][<object>])."""

from __future__ import annotations

import json
from pathlib import Path

STYLE_KEYS = {"shape", "colour", "alpha"}   # only change the drawing, never the fit


def _clean(d: dict) -> dict:
    return {k: v for k, v in (d or {}).items() if not k.startswith("_")}


def object_settings(config: dict, name: str, top: dict, hsv: dict, mesh: dict) -> dict:
    """config["arena_plotting"][name] with defaults filled in: top-level keys, "hsv", "mesh"."""
    user = (config.get("arena_plotting") or {}).get(name) or {}
    return {**top, **_clean({k: v for k, v in user.items() if k not in ("hsv", "mesh")}),
            "hsv": {**hsv, **_clean(user.get("hsv"))},
            "mesh": {**mesh, **_clean(user.get("mesh"))}}


def cache_key(s: dict, views, **extra) -> dict:
    """What a fit depends on: its settings (not the style), the calibration and inputs, plus extra.
    Round-tripped through JSON so it compares equal to the copy read back from the cache."""
    key = {"settings": {**s, "mesh": {k: v for k, v in s["mesh"].items() if k not in STYLE_KEYS}},
           "cameras": {v.name: {"input": v.input_name, "K": v.camera.K.tolist(), "dist": v.camera.dist.tolist(),
                                "R": v.camera.R.tolist(), "t": v.camera.t.tolist()} for v in views},
           **extra}
    return json.loads(json.dumps(key))


def read_cache(path: Path, key: dict):
    """(True, saved result or None) if path holds a fit for key, else (False, None)."""
    if not path.is_file():
        return False, None
    saved = json.loads(path.read_text())
    return (True, saved.get("result")) if saved.get("key") == key else (False, None)


def write_cache(path: Path, key: dict, result: dict | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": key, "result": result}, indent=1))
