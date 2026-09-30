"""Evaluate one or more detectors against a ground-truth dataset.

Pipeline per model:
    1. Build a cache key from the detector parameters (incl. weights sha256) and
       the dataset fingerprint (which frames are evaluated).
    2. If a previous run in the output folder has predictions with the same key,
       reuse them; otherwise run inference and save them.
    3. Compute COCO-style metrics, print them, and save everything to
       output/run_<datetime>/.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from datasets import Dataset, dataset_summary, iter_frame_images, load_dataset
from detectors import Detector, build_detector
from metrics import COCO_AREA_RANGES, COCO_IOUS, evaluate_detections
from visualise import save_track_plots, save_videos

PRED_FILE = "predictions.json"
PARAMS_FILE = "prediction_params.json"


# ----------------------------------------------------------------------------- caching

def make_cache_key(detector: Detector, ds: Dataset) -> tuple[str, dict]:
    params = {"detector": detector.cache_params(), "dataset_fingerprint": ds.fingerprint()}
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16], params


def find_cached_predictions(output_root: Path, cache_key: str) -> Path | None:
    """Most recent run folder containing predictions with this cache key."""
    hits = []
    for params_path in output_root.glob(f"run_*/*/{PARAMS_FILE}"):
        try:
            if json.loads(params_path.read_text()).get("cache_key") == cache_key \
                    and (params_path.parent / PRED_FILE).is_file():
                hits.append(params_path.parent)
        except (json.JSONDecodeError, OSError):
            continue
    return max(hits, key=lambda p: p.parent.name) if hits else None


def save_predictions(path: Path, preds: dict[str, np.ndarray], class_names: dict, timing: dict) -> None:
    payload = {
        "format": "xyxy_pixels_score_class",
        "class_names": {str(k): v for k, v in class_names.items()},
        "timing": timing,
        "predictions": {
            k: [[round(float(x), 2) for x in r[:4]] + [round(float(r[4]), 5), int(r[5])] for r in v]
            for k, v in preds.items()
        },
    }
    path.write_text(json.dumps(payload, separators=(",", ":")))


def load_predictions(path: Path) -> tuple[dict[str, np.ndarray], dict, dict]:
    payload = json.loads(path.read_text())
    preds = {k: np.array(v, dtype=np.float64).reshape(-1, 6) for k, v in payload["predictions"].items()}
    class_names = {int(k): v for k, v in payload.get("class_names", {}).items()}
    return preds, class_names, payload.get("timing", {})


def run_inference(detector: Detector, ds: Dataset) -> tuple[dict[str, np.ndarray], dict]:
    detector.load()
    preds, infer_s = {}, 0.0
    t_start = time.perf_counter()
    for frame, img in tqdm(iter_frame_images(ds.frames), total=len(ds.frames),
                           desc=f"Inference [{detector.name}]", unit="frame"):
        t0 = time.perf_counter()
        preds[frame.key] = detector.predict(img)
        infer_s += time.perf_counter() - t0
    wall = time.perf_counter() - t_start
    n = max(len(ds.frames), 1)
    timing = {
        "frames": len(ds.frames),
        "inference_seconds": round(infer_s, 2),
        "wall_seconds": round(wall, 2),
        "ms_per_frame": round(1000 * infer_s / n, 2),
        "fps": round(n / infer_s, 2) if infer_s else None,
    }
    return preds, timing


def get_predictions(detector: Detector, ds: Dataset, output_root: Path, model_dir: Path,
                    force: bool) -> tuple[dict[str, np.ndarray], dict]:
    cache_key, params = make_cache_key(detector, ds)
    cached_dir = None if force else find_cached_predictions(output_root, cache_key)

    if cached_dir:
        print(f"[{detector.name}] Reusing cached predictions from {cached_dir.relative_to(output_root)}")
        preds, class_names, timing = load_predictions(cached_dir / PRED_FILE)
        shutil.copy2(cached_dir / PRED_FILE, model_dir / PRED_FILE)
        source = str(cached_dir.relative_to(output_root))
    else:
        preds, timing = run_inference(detector, ds)
        class_names = detector.class_names
        save_predictions(model_dir / PRED_FILE, preds, class_names, timing)
        # Evaluate the saved (rounded) values so fresh and cached runs give identical metrics.
        preds, _, _ = load_predictions(model_dir / PRED_FILE)
        source = "inference"

    info = {
        "cache_key": cache_key,
        "prediction_source": source,
        "model_config": detector.cfg,
        "cache_params": params,
        "class_names": {str(k): v for k, v in class_names.items()},
        "timing": timing,
    }
    (model_dir / PARAMS_FILE).write_text(json.dumps(info, indent=2, default=str))
    return preds, info


# ----------------------------------------------------------------------------- reporting

def _fmt(v: float) -> str:
    return "  n/a " if v is None or v < 0 else f"{v:.4f}"


def format_model_report(name: str, info: dict, results: dict, class_names: dict) -> str:
    cfg = info["model_config"]
    timing = info["timing"]
    lines = [
        "=" * 78,
        f" MODEL: {name}",
        "=" * 78,
        f"  type/model_type : {cfg.get('type', 'sahi')} / {cfg.get('model_type', '-')}",
        f"  weights         : {cfg.get('model_path')}",
        f"  conf threshold  : {cfg.get('confidence_threshold')}",
    ]
    if cfg.get("type", "sahi") == "sahi":
        lines.append(
            f"  slicing         : {cfg.get('slice_width')}x{cfg.get('slice_height')}, overlap "
            f"{cfg.get('overlap_width_ratio')}/{cfg.get('overlap_height_ratio')}, "
            f"post {cfg.get('postprocess_type')}/{cfg.get('postprocess_match_metric')}"
            f"@{cfg.get('postprocess_match_threshold')}, full-frame pass={cfg.get('perform_standard_pred')}"
        )
    lines += [
        f"  predictions     : {info['prediction_source']}  (cache key {info['cache_key']})",
        f"  speed           : {timing.get('ms_per_frame')} ms/frame, {timing.get('fps')} FPS "
        f"over {timing.get('frames')} frames",
        f"  total dets      : {results['settings']['predictions']}",
        "",
        "  Mean average precision (COCO 101-pt, maxDets="
        f"{results['settings']['max_dets_per_image']})",
    ]
    areas = list(next(iter(results["mAP"].values())).keys())
    lines.append("  " + f"{'':<14}" + "".join(f"{a:>10}" for a in areas))
    for map_name, by_area in results["mAP"].items():
        lines.append("  " + f"{map_name:<14}" + "".join(f"{_fmt(by_area[a]):>10}" for a in areas))

    lines += ["", "  Precision / recall (all sizes)"]
    lines.append("  " + f"{'':<22}{'conf':>7}{'P':>9}{'R':>9}{'F1':>9}{'TP':>8}{'FP':>8}{'FN':>8}")
    for iou_name, d in results["precision_recall"].items():
        for label, key in (("at conf", "at_conf"), ("best F1", "best_f1")):
            r = d[key]
            lines.append(
                "  " + f"{iou_name + ' ' + label:<22}{r['conf']:>7.3f}{r['precision']:>9.4f}"
                f"{r['recall']:>9.4f}{r['f1']:>9.4f}{r['tp']:>8}{r['fp']:>8}{r['fn']:>8}"
            )

    if len(results["per_class"]) > 1:
        lines += ["", "  Per-class AP (all sizes)"]
        map_names = list(results["mAP"])
        lines.append("  " + f"{'class':<20}" + "".join(f"{m:>14}" for m in map_names))
        for c, v in results["per_class"].items():
            label = class_names.get(int(c), c)
            lines.append("  " + f"{str(label):<20}" + "".join(f"{_fmt(v[m]['all']):>14}" for m in map_names))
    return "\n".join(lines)


def format_comparison(all_results: dict) -> str:
    """One-line-per-model summary for quick side-by-side comparison."""
    if len(all_results) < 2:
        return ""
    first = next(iter(all_results.values()))["precision_recall"]
    pr_key = "IoU@0.5" if "IoU@0.5" in first else list(first)[-1]
    tag = pr_key.replace("IoU@", "@")
    cols = [(f"P{tag}", lambda r: r["precision_recall"][pr_key]["at_conf"]["precision"]),
            (f"R{tag}", lambda r: r["precision_recall"][pr_key]["at_conf"]["recall"])]
    for m in next(iter(all_results.values()))["mAP"]:
        cols.append((m.replace("mAP", ""), lambda r, m=m: r["mAP"][m]["all"]))
    lines = ["=" * 78, " COMPARISON (all sizes)", "=" * 78,
             "  " + f"{'model':<24}" + "".join(f"{c:>10}" for c, _ in cols)]
    for name, r in all_results.items():
        lines.append("  " + f"{name:<24}" + "".join(f"{_fmt(f(r)):>10}" for _, f in cols))
    return "\n".join(lines)


def flatten_row(name: str, info: dict, results: dict) -> dict:
    row = {"model": name, "prediction_source": info["prediction_source"],
           "cache_key": info["cache_key"], "ms_per_frame": info["timing"].get("ms_per_frame")}
    for iou_name, d in results["precision_recall"].items():
        for kind in ("at_conf", "best_f1"):
            for k in ("conf", "precision", "recall", "f1", "tp", "fp", "fn"):
                row[f"{iou_name}_{kind}_{k}"] = d[kind][k]
    for map_name, by_area in results["mAP"].items():
        for a, v in by_area.items():
            row[f"{map_name}_{a}"] = v
    return row


def environment_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("sahi", "ultralytics", "torch", "cv2", "numpy"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


# ----------------------------------------------------------------------------- main entry

def run_evaluation(config: dict, force_inference: bool = False, only_models: list[str] | None = None,
                   save_video: bool = False, save_track_plot: bool = False) -> Path:
    output_root = Path(config.get("output_dir", "output"))
    run_dir = output_root / f"run_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    ev = config.get("evaluation", {})
    area_ranges = ev.get("area_ranges", COCO_AREA_RANGES)
    map_ious = ev.get("map_iou_thresholds",
                      {"mAP@0.2": [0.2], "mAP@0.5": [0.5], "mAP@0.5:0.95": COCO_IOUS.tolist()})
    pr_ious = ev.get("pr_iou_thresholds", [0.2, 0.5])
    max_dets = int(ev.get("max_dets_per_image", 100))
    pr_conf = ev.get("pr_conf_threshold")
    vis = config.get("visualisation", {})

    print(f"Loading dataset '{config['dataset'].get('name')}' ...")
    ds = load_dataset(config["dataset"])
    ds_info = dataset_summary(ds, area_ranges)
    print(f"  {ds_info['frames']} frames, {ds_info['gt_boxes']} GT boxes "
          f"(by size: {ds_info['gt_boxes_by_size']})")
    gts = {f.key: f.gt for f in ds.frames}

    models = [m for m in config["models"] if m.get("enabled", True)]
    if only_models:
        models = [m for m in config["models"] if m["name"] in only_models]
    if not models:
        raise ValueError("No models selected for evaluation.")

    metadata = {
        "run": run_dir.name,
        "started": datetime.now().isoformat(timespec="seconds"),
        "environment": environment_info(),
        "dataset": ds_info,
        "evaluation_settings": {"map_iou_thresholds": map_ious, "pr_iou_thresholds": pr_ious,
                                "area_ranges": area_ranges, "max_dets_per_image": max_dets,
                                "pr_conf_threshold": pr_conf},
        "models": {},
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    all_results, reports, rows = {}, [], []
    for mcfg in models:
        detector = build_detector(mcfg)
        model_dir = run_dir / detector.name
        model_dir.mkdir(exist_ok=True)

        preds, info = get_predictions(detector, ds, output_root, model_dir, force_inference)
        class_names = ds.class_names or {int(k): v for k, v in info["class_names"].items()}

        model_conf = pr_conf if pr_conf is not None else detector.confidence_threshold
        results = evaluate_detections(gts, preds, map_iou_thresholds=map_ious,
                                      pr_iou_thresholds=pr_ious, area_ranges=area_ranges,
                                      max_dets=max_dets, pr_conf=model_conf)
        (model_dir / "metrics.json").write_text(json.dumps(results, indent=2))

        report = format_model_report(detector.name, info, results, class_names)
        print("\n" + report)
        reports.append(report)
        all_results[detector.name] = results
        rows.append(flatten_row(detector.name, info, results))
        metadata["models"][detector.name] = {k: info[k] for k in
                                             ("cache_key", "prediction_source", "model_config", "timing")}

        vis_conf = vis.get("conf_threshold") if vis.get("conf_threshold") is not None else model_conf
        if save_track_plot:
            paths = save_track_plots(ds.frames, preds, model_dir / "track_plots", detector.name, vis_conf,
                                     max_distance=vis.get("track_max_distance_px", 50),
                                     max_gap=vis.get("track_max_frame_gap", 5),
                                     background=vis.get("track_plot_background", True))
            print(f"  Track plots   : {len(paths)} saved to {model_dir / 'track_plots'}")
        if save_video:
            paths = save_videos(ds.frames, preds, model_dir / "videos", detector.name, vis_conf,
                                fps=vis.get("video_fps", 30))
            print(f"  Videos        : {len(paths)} saved to {model_dir / 'videos'}")

    comparison = format_comparison(all_results)
    if comparison:
        print("\n" + comparison)

    header = [
        f"Evaluation run : {run_dir.name}",
        f"Dataset        : {ds_info['name']}  ({ds_info['frames']} frames, {ds_info['gt_boxes']} GT boxes, "
        f"sizes {ds_info['gt_boxes_by_size']})",
        f"Size ranges    : " + ", ".join(f"{k}=[{v[0]:g}, {v[1]:g}] px^2" for k, v in area_ranges.items()),
        f"P/R conf       : {pr_conf if pr_conf is not None else 'each model confidence_threshold'}",
        "",
    ]
    (run_dir / "summary.txt").write_text("\n".join(header + reports + ([comparison] if comparison else [])) + "\n")
    metadata["finished"] = datetime.now().isoformat(timespec="seconds")
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    (run_dir / "metrics.json").write_text(json.dumps(all_results, indent=2))
    with open(run_dir / "metrics.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {run_dir.resolve()}")
    return run_dir
