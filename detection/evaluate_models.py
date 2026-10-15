"""Evaluate one or more detectors against a ground-truth dataset.

Pipeline per model:
    1. Build a cache key from the detector parameters (incl. weights sha256) and
       the dataset fingerprint (which frames are evaluated).
    2. If a previous run in the output folder has predictions with the same key,
       reuse them; otherwise run inference and save them.
    3. Compute COCO-style metrics, print them, and save everything to
       output/run_<datetime>/.

Single dataset:  run_<datetime>/<model>/...
Batch dataset:   run_<datetime>/<input name>/<model>/...  one folder per
                 input/label pair, plus run-level metrics pooled over all inputs.
Inference only:  run_<datetime>/<input name>/<model>/predictions/  (run_inference_only,
                 no ground truth or metrics; same layout as a batch run, so the 3D
                 scripts can use it).
"""

from __future__ import annotations

import csv
import json
import platform
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common.datasets import Dataset, dataset_summary, discover_batch, discover_inputs, iter_frame_images, load_dataset
from visualise.boxes.video import save_videos
from visualise.charts.evaluation import save_detection_charts
from visualise.metrics.detection import format_batch_table, format_comparison, format_model_report, headline
from visualise.plots2d.track_plots import save_track_plots

from .detectors import Detector, build_detector
from .metrics import COCO_AREA_RANGES, COCO_IOUS, evaluate_detections
from .predictions import (PARAMS_FILE, PRED_DIR, find_cached_predictions, load_predictions, make_cache_key,
                          save_predictions)


# ----------------------------------------------------------------------------- caching

def run_inference(detector: Detector, ds: Dataset, leave: bool = True) -> tuple[dict[str, np.ndarray], dict]:
    detector.ensure_loaded()
    preds, infer_s = {}, 0.0
    t_start = time.perf_counter()
    for frame, img in tqdm(iter_frame_images(ds.frames), total=len(ds.frames),
                           desc=f"Inference [{detector.name}]", unit="frame", leave=leave,
                           dynamic_ncols=True):
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
                    force: bool, leave: bool = True) -> tuple[dict[str, np.ndarray], dict]:
    cache_key, params = make_cache_key(detector, ds)
    cached_dir = None if force else find_cached_predictions(output_root, cache_key)

    if cached_dir:
        tqdm.write(f"  [{detector.name}] Reusing cached predictions from {cached_dir.relative_to(output_root)}")
        cached = json.loads((cached_dir / PARAMS_FILE).read_text())
        class_names = {int(k): v for k, v in cached.get("class_names", {}).items()}
        timing = cached.get("timing", {})
        shutil.copytree(cached_dir / PRED_DIR, model_dir / PRED_DIR, dirs_exist_ok=True)
        preds = load_predictions(model_dir / PRED_DIR, ds)
        source = str(cached_dir.relative_to(output_root))
    else:
        preds, timing = run_inference(detector, ds, leave=leave)
        class_names = detector.class_names
        save_predictions(model_dir / PRED_DIR, ds, preds, class_names)
        # Evaluate the saved (rounded) values so fresh and cached runs give identical metrics.
        preds = load_predictions(model_dir / PRED_DIR, ds)
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
# (the text reports themselves are in visualise/metrics/detection.py)

def flatten_row(name: str, info: dict, results: dict) -> dict:
    row = {"model": name, "prediction_source": info["prediction_source"],
           "cache_key": info["cache_key"], "ms_per_frame": info["timing"].get("ms_per_frame")}
    for iou_name, d in results["precision_recall"].items():
        for kind in ("at_conf", "best_f1"):
            for k in ("conf", "precision", "recall", "f1", "tp", "fp", "fn"):
                row[f"{iou_name}_{kind}_{k}"] = d[kind][k]
    for map_name, by_area in results["average_precision"].items():
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


# ----------------------------------------------------------------------------- per dataset

def eval_settings(config: dict) -> dict:
    ev = config.get("evaluation", {})
    return {
        "area_ranges": ev.get("area_ranges", COCO_AREA_RANGES),
        "map_iou_thresholds": ev.get("map_iou_thresholds",
                                     {"mAP@0.2": [0.2], "mAP@0.5": [0.5], "mAP@0.5:0.95": COCO_IOUS.tolist()}),
        "pr_iou_thresholds": ev.get("pr_iou_thresholds", [0.2, 0.5]),
        "max_dets_per_image": int(ev.get("max_dets_per_image", 100)),
        "pr_conf_threshold": ev.get("pr_conf_threshold", 0.25),
    }


def model_conf(detector: Detector, es: dict) -> float:
    """Confidence for P/R/F1: evaluation.pr_conf_threshold; null = every saved prediction."""
    return es["pr_conf_threshold"] if es["pr_conf_threshold"] is not None else detector.confidence_threshold


def evaluate(gts: dict, preds: dict, conf: float, es: dict) -> dict:
    return evaluate_detections(gts, preds, map_iou_thresholds=es["map_iou_thresholds"],
                               pr_iou_thresholds=es["pr_iou_thresholds"], area_ranges=es["area_ranges"],
                               max_dets=es["max_dets_per_image"], pr_conf=conf)


def evaluate_dataset(ds: Dataset, detectors: list[Detector], out_dir: Path, output_root: Path, es: dict,
                     vis: dict, force_inference: bool, save_video: bool, save_track_plot: bool,
                     batch: bool = False) -> dict[str, dict]:
    """Run/load predictions for every detector on one dataset, score them and write
    out_dir/<model>/. Returns {model: {"info", "results", "preds", "report"}}."""
    gts = {f.key: f.gt for f in ds.frames}
    out = {}
    for detector in detectors:
        model_dir = out_dir / detector.name
        model_dir.mkdir(parents=True, exist_ok=True)

        preds, info = get_predictions(detector, ds, output_root, model_dir, force_inference, leave=not batch)
        class_names = ds.class_names or {int(k): v for k, v in info["class_names"].items()}
        conf = model_conf(detector, es)
        results = evaluate(gts, preds, conf, es)
        (model_dir / "metrics.json").write_text(json.dumps(results, indent=2))

        report = format_model_report(detector.name, info, results, class_names)
        tqdm.write(headline(detector.name, info, results) if batch else "\n" + report)
        out[detector.name] = {"info": info, "results": results, "preds": preds, "report": report}

        vis_conf = vis.get("conf_threshold") if vis.get("conf_threshold") is not None else conf
        if save_track_plot:
            paths = save_track_plots(ds.frames, preds, model_dir / "track_plots", detector.name, vis_conf,
                                     max_distance=vis.get("track_max_distance_px", 50),
                                     max_gap=vis.get("track_max_frame_gap", 5),
                                     background=vis.get("track_plot_background", True))
            if not batch:
                print(f"  Track plots   : {len(paths)} saved to {model_dir / 'track_plots'}")
        if save_video:
            paths = save_videos(ds.frames, preds, model_dir / "videos", detector.name, vis_conf,
                                fps=vis.get("video_fps", 30))
            if not batch:
                print(f"  Videos        : {len(paths)} saved to {model_dir / 'videos'}")
    return out


def write_results(out_dir: Path, header: list[str], model_out: dict[str, dict], extra: list[str] = (),
                  per_input: dict[str, dict] | None = None) -> None:
    """summary.txt, metrics.json, metrics.csv and summary.png for one dataset (or the pooled batch;
    per_input = {input: {model: results}} adds a per-input panel to the chart)."""
    reports = [m["report"] for m in model_out.values()]
    all_results = {name: m["results"] for name, m in model_out.items()}
    comparison = format_comparison(all_results)
    (out_dir / "summary.txt").write_text(
        "\n".join(header + reports + ([comparison] if comparison else []) + list(extra)) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(all_results, indent=2))
    write_csv(out_dir / "metrics.csv", [flatten_row(n, m["info"], m["results"]) for n, m in model_out.items()])
    save_detection_charts(out_dir / "summary.png", all_results, per_input, title=header[0] if header else "")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summary_header(run_name: str, ds_info: dict, es: dict) -> list[str]:
    pr_conf = es["pr_conf_threshold"]
    return [
        f"Evaluation run : {run_name}",
        f"Dataset        : {ds_info['name']}  ({ds_info['frames']} frames, {ds_info['gt_boxes']} GT boxes, "
        f"sizes {ds_info['gt_boxes_by_size']})",
        f"Size ranges    : " + ", ".join(f"{k}=[{v[0]:g}, {v[1]:g}] px^2" for k, v in es["area_ranges"].items()),
        f"P/R/F1 conf    : {pr_conf if pr_conf is not None else 'every saved box'}  (evaluation.pr_conf_threshold, the operating point)",
        "AP             : over every saved box (each model's inference confidence_threshold and up), "
        "i.e. the whole precision-recall curve",
        "",
    ]


def _model_meta(info: dict) -> dict:
    return {k: info[k] for k in ("cache_key", "prediction_source", "model_config", "timing")}


# ----------------------------------------------------------------------------- main entry

def select_detectors(config: dict, only_models: list[str] | None) -> list[Detector]:
    models = [m for m in config["models"] if m.get("enabled", True)]
    if only_models:
        models = [m for m in config["models"] if m["name"] in only_models]
    if not models:
        raise ValueError("No models selected for evaluation.")
    return [build_detector(m) for m in models]


def run_evaluation(config: dict, force_inference: bool = False, only_models: list[str] | None = None,
                   save_video: bool = False, save_track_plot: bool = False) -> Path:
    output_root = Path(config.get("output_dir", "output"))
    detectors = select_detectors(config, only_models)  # fail fast before creating a run folder
    run_dir = output_root / f"run_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    es = eval_settings(config)
    metadata = {
        "run": run_dir.name,
        "started": datetime.now().isoformat(timespec="seconds"),
        "environment": environment_info(),
        "evaluation_settings": es,
    }
    args = (detectors, output_root, es, config.get("visualisation", {}), force_inference, save_video,
            save_track_plot)
    if config["dataset"].get("type") == "batch":
        run_batch(config["dataset"], run_dir, metadata, *args)
    else:
        run_single(config["dataset"], run_dir, metadata, *args)

    metadata["finished"] = datetime.now().isoformat(timespec="seconds")
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"\nSaved results to {run_dir.resolve()}")
    return run_dir


def run_single(ds_cfg: dict, run_dir: Path, metadata: dict, detectors: list[Detector], output_root: Path,
               es: dict, vis: dict, force_inference: bool, save_video: bool, save_track_plot: bool) -> None:
    print(f"Loading dataset '{ds_cfg.get('name')}' ...")
    ds = load_dataset(ds_cfg)
    ds_info = dataset_summary(ds, es["area_ranges"])
    print(f"  {ds_info['frames']} frames, {ds_info['gt_boxes']} GT boxes "
          f"(by size: {ds_info['gt_boxes_by_size']})")
    metadata["dataset"] = ds_info

    out = evaluate_dataset(ds, detectors, run_dir, output_root, es, vis, force_inference, save_video,
                           save_track_plot)
    metadata["models"] = {name: _model_meta(m["info"]) for name, m in out.items()}
    comparison = format_comparison({n: m["results"] for n, m in out.items()})
    if comparison:
        print("\n" + comparison)
    write_results(run_dir, summary_header(run_dir.name, ds_info, es), out)


def pooled_info(detector: Detector, infos: list[dict]) -> dict:
    """Combine per-input prediction info into one entry for the pooled report."""
    frames = sum(i["timing"].get("frames", 0) for i in infos)
    infer_s = sum(i["timing"].get("inference_seconds", 0) for i in infos)
    n_inf = sum(i["prediction_source"] == "inference" for i in infos)
    return {
        "cache_key": "per input",
        "prediction_source": f"{len(infos)} inputs ({n_inf} inference, {len(infos) - n_inf} cached)",
        "model_config": detector.cfg,
        "timing": {"frames": frames, "inference_seconds": round(infer_s, 2),
                   "ms_per_frame": round(1000 * infer_s / frames, 2) if frames else None,
                   "fps": round(frames / infer_s, 2) if infer_s else None},
    }


def run_batch(batch_cfg: dict, run_dir: Path, metadata: dict, detectors: list[Detector], output_root: Path,
              es: dict, vis: dict, force_inference: bool, save_video: bool, save_track_plot: bool) -> None:
    pairs, warnings = discover_batch(batch_cfg)
    print(f"Batch '{batch_cfg.get('name', 'batch')}': {len(pairs)} input/label pairs, "
          f"{len(detectors)} model(s)")
    for p in pairs:
        print(f"  {Path(p.get('video') or p.get('images')).name}  <-  {Path(p['labels']).name}")
    for w in warnings:
        print(f"  WARNING: {w}")

    item_out, item_info, all_frames, skipped = {}, {}, [], list(warnings)
    class_names: dict = {}
    bar = tqdm(pairs, desc="Batch", unit="input", dynamic_ncols=True)
    for i, pair in enumerate(bar, 1):
        name = pair["name"]
        bar.set_postfix_str(name[-40:])
        tqdm.write(f"\n[{i}/{len(pairs)}] {name}")
        try:
            ds = load_dataset(pair)
        except Exception as e:  # one bad pair shouldn't sink the whole batch
            tqdm.write(f"  ERROR loading dataset, skipping: {e}")
            skipped.append(f"Skipping {name}: {e}")
            continue
        ds_info = dataset_summary(ds, es["area_ranges"])
        tqdm.write(f"  {ds_info['frames']} frames, {ds_info['gt_boxes']} GT boxes "
                   f"(by size: {ds_info['gt_boxes_by_size']})")

        item_dir = run_dir / name
        item_dir.mkdir(exist_ok=True)
        out = evaluate_dataset(ds, detectors, item_dir, output_root, es, vis, force_inference, save_video,
                               save_track_plot, batch=True)
        write_results(item_dir, summary_header(f"{run_dir.name} / {name}", ds_info, es), out)
        for m in out.values():
            m.pop("report")  # pooled reports are rebuilt below; keep memory to preds only
        item_out[name], item_info[name] = out, ds_info
        all_frames += ds.frames
        class_names = class_names or ds.class_names
    bar.close()
    if not item_out:
        raise RuntimeError("Every batch input failed to load; nothing was evaluated.")

    # Pool every input's frames and predictions (keys are "<input>/<frame>", so unique).
    pooled_ds = Dataset(name=batch_cfg.get("name", "batch"), frames=all_frames, class_names=class_names,
                        source=batch_cfg)
    pooled_ds_info = dataset_summary(pooled_ds, es["area_ranges"])
    gts = {f.key: f.gt for f in all_frames}
    pooled = {}
    for det in detectors:
        outs = [o[det.name] for o in item_out.values()]
        preds = {k: v for o in outs for k, v in o["preds"].items()}
        info = pooled_info(det, [o["info"] for o in outs])
        results = evaluate(gts, preds, model_conf(det, es), es)
        names = class_names or {int(k): v for k, v in outs[0]["info"]["class_names"].items()}
        report = format_model_report(f"{det.name}  [ALL {len(outs)} INPUTS POOLED]", info, results, names)
        pooled[det.name] = {"info": info, "results": results, "report": report}

    sizes = {n: (d["frames"], d["gt_boxes"]) for n, d in item_info.items()}
    sizes["ALL (pooled)"] = (pooled_ds_info["frames"], pooled_ds_info["gt_boxes"])
    table = format_batch_table({n: {m: o["results"] for m, o in out.items()} for n, out in item_out.items()},
                               {m: p["results"] for m, p in pooled.items()}, sizes)
    print("\n" + "\n\n".join(p["report"] for p in pooled.values()))
    comparison = format_comparison({n: p["results"] for n, p in pooled.items()})
    if comparison:
        print("\n" + comparison)
    print("\n" + table)
    if skipped:
        print("\nSkipped:\n" + "\n".join(f"  {s}" for s in skipped))

    header = summary_header(run_dir.name, pooled_ds_info, es)
    header[1:1] = [f"Batch inputs   : {len(item_out)} evaluated, {len(skipped)} skipped"]
    write_results(run_dir, header, pooled,
                  extra=["", table] + (["", "Skipped:"] + [f"  {s}" for s in skipped] if skipped else []),
                  per_input={n: {m: o["results"] for m, o in out.items()} for n, out in item_out.items()})
    # Run-level metrics.json / metrics.csv also carry every input's results.
    (run_dir / "metrics.json").write_text(json.dumps({
        "pooled": {n: p["results"] for n, p in pooled.items()},
        "inputs": {n: {m: o["results"] for m, o in out.items()} for n, out in item_out.items()},
    }, indent=2))
    rows = [{"input": "ALL (pooled)", **flatten_row(n, p["info"], p["results"])} for n, p in pooled.items()]
    rows += [{"input": n, **flatten_row(m, o["info"], o["results"])}
             for n, out in item_out.items() for m, o in out.items()]
    write_csv(run_dir / "metrics.csv", rows)

    metadata["dataset"] = pooled_ds_info
    metadata["skipped"] = skipped
    metadata["models"] = {n: _model_meta(p["info"]) for n, p in pooled.items()}
    metadata["inputs"] = {n: {"dataset": item_info[n], "models": {m: _model_meta(o["info"]) for m, o in out.items()}}
                          for n, out in item_out.items()}


# ----------------------------------------------------------------------------- inference only

def inference_inputs(ds_cfg: dict, inputs: str | None = None) -> list[dict]:
    """Per-input dataset configs, without labels: every input in `inputs` (a folder of videos /
    frame folders) if given, else in the config's dataset (a batch's inputs, or its one video /
    image folder). Batch-level settings such as label_template and start_index are kept."""
    cfg = {k: v for k, v in ds_cfg.items() if k != "labels"}
    if inputs:
        return discover_inputs({**cfg, "inputs": inputs, "items": None})
    if cfg.get("type") == "batch":
        return discover_inputs(cfg)
    source = Path(cfg.get("video") or cfg["images"])
    return [{**cfg, "name": source.stem if source.is_file() else source.name}]


def run_inference_only(config: dict, force_inference: bool = False, only_models: list[str] | None = None,
                       inputs: str | None = None) -> Path:
    """Predictions for every input, no evaluation. Cached predictions (from evaluation or
    inference runs, same detector and frames) are reused."""
    output_root = Path(config.get("output_dir", "output"))
    detectors = select_detectors(config, only_models)  # fail fast before creating a run folder
    items = inference_inputs(config["dataset"], inputs)
    if not items:
        raise SystemExit(f"No videos or frame folders found in {inputs or config['dataset'].get('inputs')}")
    run_dir = output_root / f"run_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    metadata = {"run": run_dir.name, "kind": "inference", "started": datetime.now().isoformat(timespec="seconds"),
                "environment": environment_info(), "inputs": {}, "skipped": []}

    print(f"Inference on {len(items)} input(s), {len(detectors)} model(s)")
    for i, item in enumerate(items, 1):
        name = item["name"]
        print(f"\n[{i}/{len(items)}] {name}")
        try:
            ds = load_dataset(item)
        except Exception as e:  # one bad input shouldn't sink the whole run
            print(f"  ERROR loading input, skipping: {e}")
            metadata["skipped"].append(f"Skipping {name}: {e}")
            continue
        models = {}
        for det in detectors:
            model_dir = run_dir / name / det.name
            model_dir.mkdir(parents=True, exist_ok=True)
            preds, info = get_predictions(det, ds, output_root, model_dir, force_inference)
            n = sum(len(p) for p in preds.values())
            print(f"    {det.name:<28} {len(ds.frames)} frames, {n} detections  "
                  f"[{info['timing'].get('ms_per_frame')} ms/frame, "
                  f"{'cached' if info['prediction_source'] != 'inference' else 'inference'}]")
            models[det.name] = _model_meta(info)
        metadata["inputs"][name] = {"dataset": dataset_summary(ds, {}), "models": models}

    metadata["finished"] = datetime.now().isoformat(timespec="seconds")
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    if metadata["skipped"]:
        print("\nSkipped:\n" + "\n".join(f"  {s}" for s in metadata["skipped"]))
    print(f"\nSaved predictions to {run_dir.resolve()}")
    return run_dir
