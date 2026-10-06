"""3D tracks for hand-picked objects, guided by annotated start / end points.

1. Choose a run (as in main_tracks3d.py).
2. An annotation window opens on each camera's video in turn. Mark where each object
   of interest starts and / or ends - in any views, any mix. "continue" moves to the
   next video, "finish" on the last one runs the tracking. Annotations are saved to
   <run>/3d/annotations.json and loaded again next time, so they can be refined or
   reused (--no-gui).
3. Each annotated object gets exactly one factor-graph track: the annotations fix its
   frame range, act as trusted reprojection factors and seed it, so it can bridge
   stretches where the detector lost it. See tracking3d/guided.py.

From the IDE: edit the SETTINGS block below and press Run (or F5).
From the command line, flags override the settings:
    python main_tracks3d_guided.py                  # pick a run, annotate, track
    python main_tracks3d_guided.py --no-gui         # reuse the saved annotations
    python main_tracks3d_guided.py --source gt      # track through the ground-truth labels

Outputs go to <run>/3d/<model or ground_truth>/tracks_guided/ (same files as
main_tracks3d.py, plus the annotations drawn in the plots).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate2d.config import load_config
from gui import run_annotation_gui, view_input_from_source
from tracking3d.guided import anchor_residuals, build_guided_tracks
from tracking3d.multiview import ground_truth_tracks, open_session
from tracking3d.report import write_track_outputs
from tracking3d.triangulate import MatchSettings, camera_baseline_report, triangulate_session
from tracking3d.visualise import plot_tracks_3d, save_reprojection_plots, use_interactive_backend

HERE = Path(__file__).resolve().parent

# ============================================================================ SETTINGS
CONFIG_PATH = HERE / "config.json"   # needs "cameras" (calibration); "tracking3d" tunes the graph
RUN = None           # run folder name or path; None = choose in the terminal
MODEL = None         # model in that run; None = the only one, or choose in the terminal
SOURCE = "pred"      # "pred" = the run's predictions, "gt" = its ground-truth labels
ANNOTATIONS = None   # annotations file; None = <run>/3d/annotations.json
GUI = True           # open the annotator (pre-filled with saved annotations); False = use the file as is
GT_OVERLAY = True    # draw the run's ground truth (triangulated + linked, no factor graph) in green
SHOW = True          # open an interactive (rotatable) 3D window after saving
# =====================================================================================


def annotate(session, path: Path, use_gui: bool) -> list[dict]:
    saved = json.loads(path.read_text())["annotations"] if path.is_file() else []
    if not use_gui:
        if not saved:
            raise SystemExit(f"No saved annotations at {path}. Run with the GUI first.")
        print(f"\nUsing {len(saved)} saved annotations from {path}")
        return saved

    views = []
    for v in session.views:
        detections = {f - v.camera.frame_offset: d for f, d in v.raw.items()}  # back to label frame numbers
        try:
            views.append(view_input_from_source(v.name, v.input_name, v.source, detections))
        except FileNotFoundError as e:
            raise SystemExit(f"Cannot open the video for {v.name}: {e}") from None
    print(f"\nOpening the annotator ({len(views)} views"
          f"{f', {len(saved)} saved annotations loaded' if saved else ''}) ...")
    what = "ground-truth labels" if session.source == "gt" else f"{session.model} detections"
    result = run_annotation_gui(views, saved, title=f"{session.run.name}  |  {session.label}",
                                detections_label=what)
    if result is None:
        raise SystemExit("Annotation cancelled - nothing saved, nothing tracked.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"run": session.run.name, "views": [v.name for v in session.views],
                                "annotations": result}, indent=2))
    print(f"Saved {len(result)} annotations to {path}")
    return result


def describe_annotations(annotations: list[dict], view_names: list[str]) -> list[str]:
    lines = []
    for obj in sorted({a["object"] for a in annotations}):
        parts = []
        for name in view_names:
            here = {a["kind"]: a["frame"] for a in annotations if a["object"] == obj and a["view"] == name}
            if here:
                parts.append(f"{name} " + "/".join(f"{k} {here[k]}" for k in ("start", "end") if k in here))
        lines.append(f"  #{obj}: " + ("; ".join(parts) or "-"))
    return lines


def main():
    parser = argparse.ArgumentParser(description="3D tracks for annotated objects (start / end guided factor graph).")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--run", default=RUN)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--source", choices=["pred", "gt"], default=SOURCE)
    parser.add_argument("--conf", type=float, default=None, help="Override tracking3d.conf_threshold.")
    parser.add_argument("--annotations", default=ANNOTATIONS, help="Annotations file (default <run>/3d/annotations.json).")
    parser.add_argument("--gui", action=argparse.BooleanOptionalAction, default=GUI,
                        help="Open the annotator; --no-gui reuses the saved annotations.")
    parser.add_argument("--gt-overlay", action=argparse.BooleanOptionalAction, default=GT_OVERLAY,
                        help="Overlay ground-truth 3D tracks in green (ignored with --source gt).")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=SHOW)
    args = parser.parse_args()

    config = load_config(args.config)
    session = open_session(config, args.run, args.model, args.source, args.conf)
    views, s = session.views, session.settings
    cams = [v.camera for v in views]
    print("\nCamera rig:\n" + "\n".join(camera_baseline_report(cams)))

    ann_path = Path(args.annotations) if args.annotations else session.run.path / "3d" / "annotations.json"
    annotations = annotate(session, ann_path, args.gui)
    if not annotations:
        raise SystemExit("No annotations - nothing to track.")
    print("\nAnnotated objects:\n" + "\n".join(describe_annotations(annotations, [v.name for v in views])))

    points = triangulate_session(cams, [v.dets for v in views],
                                 MatchSettings(s["max_epipolar_px"], s["max_reprojection_px"]))
    print(f"\nTriangulated {sum(len(p) for p in points.values())} points in {len(points)} frames")

    tracks, objects, warnings = build_guided_tracks(views, points, annotations, s)
    by_id = {o.id: o for o in objects}
    extra = ["Annotations"] + describe_annotations(annotations, [v.name for v in views]) + ["", "Guidance"]
    anchors_3d, per_object = [], {}
    for t in tracks:
        obj = by_id[t.id]
        res = anchor_residuals(t, obj, views)
        extra.append(f"  #{t.id}: {len(obj.anchors)} annotation(s), track meets them within "
                     f"{np.median(res):.1f} px (median), {max(res):.1f} px (max)")
        extra += [f"       {n}" for n in obj.notes]
        per_object[t.id] = {"annotations": [a.raw for a in obj.anchors], "anchor_residual_px": res,
                            "notes": obj.notes}
        for a in obj.anchors:
            anchors_3d.append((t.id, a.kind, t.xyz[a.frame - t.frames[0]]))
    if warnings:
        extra += ["", "Warnings"] + [f"  {w}" for w in warnings]
    if not tracks:
        raise SystemExit("\n".join(["No object could be tracked."] + warnings))

    out_dir = session.out_dir("tracks_guided")
    report = write_track_outputs(session, tracks, s, out_dir, f"{len(tracks)} annotated object(s), by object number",
                                 extra, {"annotation_file": str(ann_path), "objects": per_object})
    print("\n" + report)

    fruit = np.asarray(s["fruit_xyz"], float) if s.get("fruit_xyz") else None
    pts = np.array([[f, *p.xyz, len(p.views)] for f, ps in points.items() for p in ps]).reshape(-1, 5)
    raw_ann = {}
    for a in annotations:
        raw_ann.setdefault(a["view"], []).append((a["object"], a["kind"], a["x"], a["y"]))
    reproj = save_reprojection_plots(tracks, views, out_dir / "reprojection", annotations=raw_ann)
    title = (f"{session.run.name}  |  {session.label}  |  {len(tracks)} guided tracks "
             f"(+ = annotated start, x = annotated end)")
    gt = ground_truth_tracks(session, config["cameras"]) if args.gt_overlay and args.source == "pred" else None
    if gt:
        title += "  |  green = ground truth"
    show = args.show and use_interactive_backend()
    plot_tracks_3d(tracks, cams, out_dir / "tracks_3d.png", title, pts, fruit, gt, anchors_3d, close=not show)
    print(f"\nSaved to {out_dir}  (tracks.csv, tracks.json, summary.txt, tracks_3d.png, "
          f"{len(reproj)} reprojection plots)")
    if show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
