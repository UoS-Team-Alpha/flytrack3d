"""Factor-graph tracks evaluated against the run's ground truth, in 3D and back in 2D.

3D: every track position is matched, frame by frame, to the ground-truth 3D points (the
    run's GT labels triangulated in 2+ views and linked - the green overlay) with the
    Hungarian method on distance, within eval_match_distance. Those points are
    triangulated themselves, so they carry a small error of their own.

2D: each track is projected back into every camera (full camera model, with distortion).
    In frames where the track has no detection in a view - the detector missed the object,
    or it fell outside the association gate - a box is inserted at the projection, with
    score backproj_score. Its size is the object's physical size (box size x depth / focal
    length, from the track's own detections in that view, interpolated in time; the median
    over all views if this view has none) projected at the track's depth in that frame, so
    it shrinks and grows as the object moves away and back. The run's 2D metrics are then
    recomputed (same settings as main_evaluate2d.py) for three prediction sets:
        detector         the saved predictions, as evaluated by main_evaluate2d.py
        + factor graph   plus the inserted boxes
        on tracks only   only the predictions a track used, plus the inserted boxes
                         (everything the factor graph rejected is dropped)

With visualise_backproj_graph or visualise_backproj_video, the inserted boxes are also saved as
YOLO labels, laid out like the run's predictions (one folder per input, one .txt per frame,
"class xc yc w h score", plus data.yaml): reprojection/labels/<input>/<sequence>/<frame>.txt.
visualise_backproj_video renders reprojection/<camera>.mp4 per camera: detector boxes in blue,
ground truth in green, every track's reprojected position as a red dot (also where the detector
has a box, to show how well the track follows it) and the inserted boxes in red.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from evaluate2d.datasets import Dataset, load_dataset, read_class_names
from evaluate2d.evaluate_models import (PRED_DIR, _fmt, _pr_key, eval_settings, evaluate, load_predictions,
                                        save_predictions)
from evaluate2d.visualise import _safe_name, frame_number, save_track_plots

from .factor_graph import Track3D
from .multiview import View
from .visualise import save_backproj_video

VARIANTS = ("detector", "+ factor graph", "on tracks only")


# ----------------------------------------------------------------------------- 3D

def _error_stats(err: np.ndarray) -> dict:
    """err: (N, 3) track - ground truth."""
    d = np.linalg.norm(err, axis=1)
    if not len(d):
        return {"n": 0}
    return {"n": int(len(d)), "rmse": float(np.sqrt((d ** 2).mean())), "mean": float(d.mean()),
            "median": float(np.median(d)), "p95": float(np.percentile(d, 95)), "max": float(d.max()),
            "rmse_xyz": np.sqrt((err ** 2).mean(axis=0)).tolist()}


def evaluate_3d(tracks: list[Track3D], gt_tracks: list[np.ndarray], gate: float) -> dict:
    """gt_tracks: (K, 4) frame, x, y, z arrays (multiview.ground_truth_tracks)."""
    gt: dict[int, list] = {}
    for g, t in enumerate(gt_tracks, 1):
        for row in t:
            gt.setdefault(int(row[0]), []).append((g, row[1:4]))
    pred: dict[int, list] = {}
    for t in tracks:
        for f, p, n in zip(t.frames, t.xyz, t.views):
            pred.setdefault(int(f), []).append((t.id, p, int(n)))

    matches, in_gt_frames = [], 0   # (track id, gt id, error xyz, views of the track in that frame)
    for f, ps in pred.items():
        gs = gt.get(f)
        if not gs:
            continue
        in_gt_frames += len(ps)
        cost = np.linalg.norm(np.array([p[1] for p in ps])[:, None] - np.array([g[1] for g in gs])[None], axis=2)
        for r, c in zip(*linear_sum_assignment(cost)):
            if cost[r, c] <= gate:
                matches.append((ps[r][0], gs[c][0], ps[r][1] - gs[c][1], ps[r][2]))

    def stats(ms):
        return _error_stats(np.array([m[2] for m in ms]).reshape(-1, 3))

    n_gt = sum(len(v) for v in gt.values())
    per_track = []
    for t in tracks:
        ms = [m for m in matches if m[0] == t.id]
        main, n_main = Counter(m[1] for m in ms).most_common(1)[0] if ms else (None, 0)
        per_track.append({"id": t.id, "frames": len(t.frames), "matched": len(ms), "gt_track": main,
                          "matched_to_gt_track": n_main, "error": stats(ms)})
    per_gt = []
    for g, t in enumerate(gt_tracks, 1):
        ms = [m for m in matches if m[1] == g]
        per_gt.append({"id": g, "start_frame": int(t[0, 0]), "end_frame": int(t[-1, 0]), "points": len(t),
                       "matched": len(ms), "tracks": sorted({m[0] for m in ms})})
    return {
        "match_distance": gate, "gt_tracks": len(gt_tracks), "gt_points": n_gt,
        "track_points": sum(len(t.frames) for t in tracks), "track_points_in_gt_frames": in_gt_frames,
        "matched": len(matches),
        "recall": len(matches) / n_gt if n_gt else None,
        "precision": len(matches) / in_gt_frames if in_gt_frames else None,
        "error": stats(matches),
        "error_by_views": {"0-1 views": stats([m for m in matches if m[3] <= 1]),
                           "2 views": stats([m for m in matches if m[3] == 2]),
                           "3 views": stats([m for m in matches if m[3] >= 3])},
        "per_track": per_track, "per_gt_track": per_gt,
    }


def _stats_row(label: str, e: dict) -> str:
    if not e["n"]:
        return f"  {label:<12}{0:>7}"
    return (f"  {label:<12}{e['n']:>7}{e['rmse']:>9.2f}{e['mean']:>8.2f}{e['median']:>8.2f}{e['p95']:>8.2f}"
            f"{e['max']:>8.2f}   " + " / ".join(f"{v:.2f}" for v in e["rmse_xyz"]))


def format_3d(r: dict) -> str:
    pct = lambda v: f"{100 * v:.1f}%" if v is not None else "n/a"  # noqa: E731
    lines = ["3D evaluation against ground truth (world units = mm with the calibration in mm)", "",
             f"  ground truth : {r['gt_points']} points in {r['gt_tracks']} tracks (GT labels triangulated in 2+ "
             f"views and linked - not error-free themselves)",
             f"  matching     : per frame, Hungarian on 3D distance, within {r['match_distance']:g}",
             f"  recall       : {r['matched']} / {r['gt_points']} ground-truth points matched by a track "
             f"({pct(r['recall'])})",
             f"  precision    : {r['matched']} / {r['track_points_in_gt_frames']} track points in frames with "
             f"ground truth matched ({pct(r['precision'])})",
             f"                 ({r['track_points'] - r['track_points_in_gt_frames']} track points fall in "
             f"frames without any ground-truth 3D point and are not scored)", "",
             "  Error of matched points",
             f"  {'':<12}{'n':>7}{'RMSE':>9}{'mean':>8}{'median':>8}{'p95':>8}{'max':>8}   RMSE x / y / z",
             _stats_row("all", r["error"])]
    lines += [_stats_row(k, e) for k, e in r["error_by_views"].items()]
    lines += ["  (views = cameras with a detection on the track in that frame; 0-1 = position mostly from the",
              "   motion prior and single camera rays)", "",
              "  Per track",
              f"  {'id':>4}{'frames':>8}{'matched':>9}{'GT track':>10}{'share':>8}{'RMSE':>9}{'median':>8}{'max':>8}"]
    for t in r["per_track"]:
        e = t["error"]
        share = f"{100 * t['matched_to_gt_track'] / t['matched']:.0f}%" if t["matched"] else "-"
        lines.append(f"  {t['id']:>4}{t['frames']:>8}{t['matched']:>9}{t['gt_track'] or '-':>10}{share:>8}"
                     + (f"{e['rmse']:>9.2f}{e['median']:>8.2f}{e['max']:>8.2f}" if e["n"] else ""))
    lines += ["  (GT track = the ground-truth track it matches most; share = fraction of its matches on it)", "",
              "  Per ground-truth track",
              f"  {'id':>4}{'frames':>13}{'points':>8}{'covered':>9}  tracks"]
    for g in r["per_gt_track"]:
        lines.append(f"  {g['id']:>4}{g['start_frame']:>6}-{g['end_frame']:<6}{g['points']:>8}"
                     f"{100 * g['matched'] / g['points']:>8.1f}%  {', '.join(map(str, g['tracks'])) or '-'}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- 2D

@dataclass
class ViewData:
    ds: Dataset
    preds: dict[str, np.ndarray]     # key -> (M, 6) x1, y1, x2, y2, score, class (raw pixels)
    keys: dict[int, str]             # aligned frame number -> dataset frame key
    conf: float                      # confidence for P/R (as in main_evaluate2d.py)


def load_view_data(session, view: View, es: dict) -> ViewData:
    meta = session.run.metadata["inputs"][view.input_name]
    ds = load_dataset(meta["dataset"]["source"])
    preds = load_predictions(session.run.path / view.input_name / session.model / PRED_DIR, ds)
    keys = {frame_number(f, i) + view.camera.frame_offset: f.key for i, f in enumerate(ds.frames)}
    conf = es["pr_conf_threshold"]
    if conf is None:
        conf = meta["models"][session.model]["model_config"].get("confidence_threshold", 0.0)
    return ViewData(ds, preds, keys, float(conf))


def backproject(tracks: list[Track3D], views: list[View], data: list[ViewData],
                score: float) -> tuple[list[dict], list[dict]]:
    """Per view: ({key: (M, 6) inserted boxes}, {key: bool mask of the predictions a track used})."""
    inserted: list[dict] = [{} for _ in views]
    on_track: list[dict] = [{} for _ in views]
    for t in tracks:
        observed, sizes = set(), [[] for _ in views]   # sizes: (frame, physical w, physical h, class)
        for f, v, j in t.observations:
            key = data[v].keys.get(f)
            boxes = data[v].preds.get(key, np.zeros((0, 6)))
            if not len(boxes):
                continue
            # The track's detection is a box centre; find its box in the saved predictions.
            dist = np.linalg.norm((boxes[:, :2] + boxes[:, 2:4]) / 2 - views[v].raw[f][j, :2], axis=1)
            i = int(dist.argmin())
            if dist[i] > 0.5:
                continue
            observed.add((v, f))
            on_track[v].setdefault(key, np.zeros(len(boxes), bool))[i] = True
            cam, b = views[v].camera, boxes[i]
            z = cam.depth(t.xyz[f - t.frames[0]])[0]
            sizes[v].append((f, (b[2] - b[0]) * z / cam.K[0, 0], (b[3] - b[1]) * z / cam.K[1, 1], b[5]))
        everywhere = np.array([s for per_view in sizes for s in per_view]).reshape(-1, 4)
        if not len(everywhere):
            continue   # no detection to take the object's size from
        cls = int(np.bincount(everywhere[:, 3].astype(int)).argmax())
        for v, view in enumerate(views):
            cam, (w, h) = view.camera, view.size
            s = np.array(sorted(sizes[v])).reshape(-1, 4)
            depth, uv = cam.depth(t.xyz), cam.project(t.xyz, distort=True)
            for k, f in enumerate(t.frames):
                f = int(f)
                key = data[v].keys.get(f)
                x, y = uv[k]
                if key is None or (v, f) in observed or depth[k] <= 0 or not (0 <= x < w and 0 <= y < h):
                    continue
                if len(s):
                    pw, ph = np.interp(f, s[:, 0], s[:, 1]), np.interp(f, s[:, 0], s[:, 2])
                else:
                    pw, ph = np.median(everywhere[:, 1]), np.median(everywhere[:, 2])
                bw, bh = pw * cam.K[0, 0] / depth[k], ph * cam.K[1, 1] / depth[k]
                inserted[v].setdefault(key, []).append([x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2, score, cls])
    return [{k: np.array(b) for k, b in ins.items()} for ins in inserted], on_track


def project_tracks(tracks: list[Track3D], views: list[View], data: list[ViewData]) -> list[dict]:
    """Per view: {frame key: (N, 2) x, y} of every track projected into the image (full camera
    model, with distortion), in every frame it is in front of the camera and inside the image."""
    points: list[dict] = [{} for _ in views]
    for t in tracks:
        for v, view in enumerate(views):
            (w, h), cam = view.size, view.camera
            depth, uv = cam.depth(t.xyz), cam.project(t.xyz, distort=True)
            for f, d, (x, y) in zip(t.frames, depth, uv):
                key = data[v].keys.get(int(f))
                if key is not None and d > 0 and 0 <= x < w and 0 <= y < h:
                    points[v].setdefault(key, []).append((x, y))
    return [{k: np.array(p) for k, p in pts.items()} for pts in points]


def _variants(d: ViewData, ins: dict, on: dict) -> dict[str, dict]:
    empty = np.zeros((0, 6))
    plus = {k: np.vstack([p, ins.get(k, empty)]) for k, p in d.preds.items()}
    only = {k: np.vstack([p[on.get(k, np.zeros(len(p), bool))], ins.get(k, empty)]) for k, p in d.preds.items()}
    return dict(zip(VARIANTS, (d.preds, plus, only)))


def evaluate_2d(views: list[View], data: list[ViewData], inserted: list[dict], on_track: list[dict],
                es: dict) -> dict:
    """{view name or "ALL (pooled)": {variant: metrics + box counts}}."""
    out, pooled_gt, pooled = {}, {}, {v: {} for v in VARIANTS}
    for view, d, ins, on in zip(views, data, inserted, on_track):
        gts = {f.key: f.gt for f in d.ds.frames}
        pooled_gt.update(gts)
        out[view.name] = {}
        for name, preds in _variants(d, ins, on).items():
            pooled[name].update(preds)
            out[view.name][name] = {**evaluate(gts, preds, d.conf, es), "boxes": int(sum(map(len, preds.values())))}
    conf = data[0].conf
    out["ALL (pooled)"] = {name: {**evaluate(pooled_gt, preds, conf, es), "boxes": int(sum(map(len, preds.values())))}
                           for name, preds in pooled.items()}
    return out


def format_2d(r: dict, data: list[ViewData], score: float) -> str:
    first = r["ALL (pooled)"]["detector"]
    pr_key = _pr_key(first)
    tag = pr_key.replace("IoU@", "@")
    maps = list(first["mAP"])
    width = max(len(n) for n in r) + 2
    lines = ["2D evaluation with the factor-graph tracks projected back into each camera", "",
             f"  detector       : the saved predictions (as main_evaluate2d.py scored them)",
             f"  + factor graph : plus a box wherever a track has no detection in that view (score {score:g})",
             f"  on tracks only : only the predictions a track used, plus those boxes",
             f"  P / R / F1 at conf {data[0].conf:g}; all sizes", "",
             f"  {'view':<{width}}{'predictions':<16}{'boxes':>7}{'P' + tag:>9}{'R' + tag:>9}{'F1' + tag:>9}"
             + "".join(f"{m.replace('mAP', ''):>10}" for m in maps)]
    for view, by_variant in r.items():
        for i, (name, res) in enumerate(by_variant.items()):
            at = res["precision_recall"][pr_key]["at_conf"]
            lines.append(f"  {view if i == 0 else '':<{width}}{name:<16}{res['boxes']:>7}{_fmt(at['precision']):>9}"
                         f"{_fmt(at['recall']):>9}{_fmt(at['f1']):>9}"
                         + "".join(f"{_fmt(res['mAP'][m]['all']):>10}" for m in maps))
        lines.append("")
    if score < data[0].conf:
        lines.append(f"  NOTE: backproj_score {score:g} is below the P/R confidence {data[0].conf:g}, so inserted "
                     f"boxes only count towards mAP.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- entry

def evaluate_tracks(session, config: dict, tracks: list[Track3D], gt_tracks: list[np.ndarray] | None,
                    out_dir: Path) -> list[str]:
    """Writes evaluation_3d.txt, evaluation_2d.txt, evaluation.json, track_plots_backproj/ and the
    back-projected labels and videos in reprojection/ (as enabled in tracking3d). Returns the lines
    to print."""
    s, msgs, results = session.settings, [], {}
    if session.source != "pred":
        return ["Evaluation skipped: tracks built from the ground-truth labels."]
    if gt_tracks:
        results["3d"] = evaluate_3d(tracks, gt_tracks, float(s["eval_match_distance"]))
        text = format_3d(results["3d"])
        (out_dir / "evaluation_3d.txt").write_text(text + "\n")
        msgs.append(text)
    else:
        msgs.append("3D evaluation skipped: no ground-truth 3D points for this run.")

    if s["evaluate_2d"] or s["visualise_backproj_graph"] or s["visualise_backproj_video"]:
        es = eval_settings(config)
        data = [load_view_data(session, v, es) for v in session.views]
        inserted, on_track = backproject(tracks, session.views, data, float(s["backproj_score"]))
        n_ins = sum(len(b) for ins in inserted for b in ins.values())
        has_gt = any(len(f.gt) for d in data for f in d.ds.frames)
        if s["evaluate_2d"] and has_gt:
            results["2d"] = evaluate_2d(session.views, data, inserted, on_track, es)
            text = format_2d(results["2d"], data, float(s["backproj_score"]))
            (out_dir / "evaluation_2d.txt").write_text(text + "\n")
            msgs.append(text)
        elif s["evaluate_2d"]:
            msgs.append("2D evaluation skipped: this run has no ground-truth labels.")
        vis = config.get("visualisation", {})
        if s["visualise_backproj_graph"] or s["visualise_backproj_video"]:
            label_dir = out_dir / "reprojection" / "labels"
            for v, d, ins in zip(session.views, data, inserted):
                pred_dir = session.run.path / v.input_name / session.model / PRED_DIR
                save_predictions(label_dir / v.input_name, d.ds, ins,
                                 d.ds.class_names or read_class_names(pred_dir / "data.yaml"))
            msgs.append(f"Back-projected labels: {label_dir}  ({n_ins} boxes inserted over all views)")
        if s["visualise_backproj_video"]:
            points = project_tracks(tracks, session.views, data)
            for v, d, ins, pts in zip(session.views, data, inserted, points):
                save_backproj_video(d.ds.frames, d.preds, ins, pts, out_dir / "reprojection" / f"{_safe_name(v.name)}.mp4",
                                    v.name, session.model, d.conf, fps=float(vis.get("video_fps", 30)))
            msgs.append(f"Back-projection videos: {out_dir / 'reprojection'}/<camera>.mp4")
        if s["visualise_backproj_graph"]:
            plot_dir = out_dir / "track_plots_backproj"
            for v, d, ins in zip(session.views, data, inserted):
                save_track_plots(d.ds.frames, d.preds, plot_dir, session.model, d.conf,
                                 max_distance=vis.get("track_max_distance_px", 50),
                                 max_gap=vis.get("track_max_frame_gap", 5),
                                 background=vis.get("track_plot_background", True),
                                 extra={k: (b[:, :2] + b[:, 2:4]) / 2 for k, b in ins.items()},
                                 extra_label="factor graph, back-projected where the detector has no box")
            msgs.append(f"Back-projected track plots: {plot_dir}  ({n_ins} boxes inserted over all views)")
    if results:
        (out_dir / "evaluation.json").write_text(json.dumps(results, indent=2))
    return msgs
