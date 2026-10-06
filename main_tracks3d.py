"""Estimate 3D tracks (scene entry -> fruit) from a previous batch evaluation run.

Triangulated points (see main_points3d.py) seed tracks, which a gtsam factor
graph then refines against every 2D detection - including those seen by a single
camera - with a constant-velocity motion prior. See tracking3d/factor_graph.py for details
and "tracking3d" in config.example.json for the settings.

From the IDE: edit the SETTINGS block below and press Run (or F5).
From the command line, flags override the settings:
    python main_tracks3d.py                     # pick a run from a list
    python main_tracks3d.py --n-tracks 3        # keep the 3 best-supported tracks
    python main_tracks3d.py --source gt         # build tracks from the ground-truth labels
    python main_tracks3d.py --no-show

Outputs go to <run>/3d/<model or ground_truth>/tracks/:
    tracks.csv               per-frame position, uncertainty and views of every track
    tracks.json / summary.txt  per-track entry point, length, speed, fruit distance
    tracks_3d.png            3D + orthographic views with the raw triangulated points
    reprojection/<cam>.png   tracks projected back into each camera (calibration check)
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate2d.config import load_config
from tracking3d.factor_graph import build_tracks
from tracking3d.multiview import ground_truth_tracks, open_session
from tracking3d.triangulate import MatchSettings, camera_baseline_report, triangulate_session
from tracking3d.visualise import plot_tracks_3d, save_reprojection_plots, use_interactive_backend

HERE = Path(__file__).resolve().parent

# ============================================================================ SETTINGS
CONFIG_PATH = HERE / "config.json"   # needs "cameras" (calibration); "tracking3d" tunes the graph
RUN = None           # run folder name or path; None = choose in the terminal
MODEL = None         # model in that run; None = the only one, or choose in the terminal
SOURCE = "pred"      # "pred" = the run's predictions, "gt" = its ground-truth labels
N_TRACKS = None      # keep the n best tracks; None = tracking3d.n_tracks (null = every valid track)
GT_OVERLAY = True    # draw the run's ground truth (triangulated + linked, no factor graph) in green
SHOW = True          # open an interactive (rotatable) 3D window after saving
# =====================================================================================


def format_summary(session, tracks, summaries, s) -> str:
    lines = [f"Session   : {session.run.name}",
             f"Detections: {session.label}",
             f"Views     : " + ", ".join(f"{v.name} <- {v.input_name}" for v in session.views),
             f"Settings  : " + ", ".join(f"{k}={v}" for k, v in s.items()), "",
             f"{len(tracks)} track(s), best supported first", ""]
    fruit = s.get("fruit_xyz") is not None
    head = f"  {'id':>3}{'frames':>13}{'len':>6}{'dets':>6}{'3-view':>8}{'2-view':>8}{'1-view':>8}" \
           f"{'reproj':>8}{'sigma':>8}{'path':>10}  {'entry xyz':<26}{'+/-':>7}{'2+ views @':>11}"
    lines.append(head + ("            fruit min @frame" if fruit else ""))
    for t in summaries:
        fv = t["frames_by_views"]
        row = (f"  {t['id']:>3}{t['start_frame']:>6}-{t['end_frame']:<6}{t['frames']:>6}{t['detections']:>6}"
               f"{fv.get(3, 0):>8}{fv.get(2, 0):>8}{fv.get(1, 0):>8}"
               f"{t['reprojection_px_median'] or 0:>8.2f}{t['sigma_median'] or float('nan'):>8.2f}"
               f"{t['path_length']:>10.1f}  {'(' + ', '.join(f'{v:.1f}' for v in t['entry_xyz']) + ')':<26}"
               f"{t['entry_sigma'] or float('nan'):>7.1f}{t['first_multiview_frame']:>11}")
        if fruit:
            row += f"  {t['fruit_closest_distance']:>10.1f} @{t['fruit_closest_frame']}"
        lines.append(row)
    lines += ["", "  frames by views: frames where the track was detected in 3 / 2 / 1 cameras "
                  "(others are bridged by the motion prior)",
              "  reproj = median px error of its detections; sigma = median 1-sigma 3D uncertainty",
              "  entry +/- = 1-sigma at the entry point (large while only one camera sees it: depth then",
              "  comes from the motion prior alone); 2+ views @ = first frame seen by two or more cameras"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="3D tracks from multi-view detections with a factor graph.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--run", default=RUN)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--source", choices=["pred", "gt"], default=SOURCE)
    parser.add_argument("--n-tracks", type=int, default=N_TRACKS,
                        help="Keep the n best-supported tracks (overrides tracking3d.n_tracks).")
    parser.add_argument("--conf", type=float, default=None, help="Override tracking3d.conf_threshold.")
    parser.add_argument("--gt-overlay", action=argparse.BooleanOptionalAction, default=GT_OVERLAY,
                        help="Overlay ground-truth 3D tracks in green (ignored with --source gt).")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=SHOW)
    args = parser.parse_args()

    config = load_config(args.config)
    session = open_session(config, args.run, args.model, args.source, args.conf)
    views, s = session.views, session.settings
    if args.n_tracks is not None:
        s["n_tracks"] = args.n_tracks
    cams = [v.camera for v in views]
    print("\nCamera rig:\n" + "\n".join(camera_baseline_report(cams)))

    points = triangulate_session(cams, [v.dets for v in views],
                                 MatchSettings(s["max_epipolar_px"], s["max_reprojection_px"]))
    n_pts = sum(len(p) for p in points.values())
    print(f"\nTriangulated {n_pts} points in {len(points)} frames")
    if not n_pts:
        raise SystemExit("No detections could be matched across views - check the calibration.")

    tracks, tracklets = build_tracks(views, points, s)
    print(f"Linked {len(tracklets)} tracklets (>= {s['min_tracklet_frames']} frames) -> {len(tracks)} tracks")
    if not tracks:
        raise SystemExit("No tracks survived. Try lowering min_tracklet_frames / min_track_frames or "
                         "raising link_max_distance (world units per frame).")

    fruit = np.asarray(s["fruit_xyz"], float) if s.get("fruit_xyz") else None
    summaries = [t.summary(fruit) for t in tracks]
    out_dir = session.out_dir("tracks")
    report = format_summary(session, tracks, summaries, s)
    print("\n" + report)
    (out_dir / "summary.txt").write_text(report + "\n")
    (out_dir / "tracks.json").write_text(json.dumps({"run": session.run.name, "detections": session.label,
                                                     "settings": s, "tracks": summaries}, indent=2))
    with open(out_dir / "tracks.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["track", "frame", "x", "y", "z", "sigma", "n_views"])
        for t in tracks:
            for f, p, sg, nv in zip(t.frames, t.xyz, t.sigma, t.views):
                w.writerow([t.id, int(f), *p.round(4), round(float(sg), 4), int(nv)])

    pts = np.array([[f, *p.xyz, len(p.views)] for f, ps in points.items() for p in ps])
    reproj = save_reprojection_plots(tracks, views, out_dir / "reprojection")
    title = f"{session.run.name}  |  {session.label}  |  {len(tracks)} tracks (triangle = entry, square = exit)"
    gt = ground_truth_tracks(session, config["cameras"]) if args.gt_overlay and args.source == "pred" else None
    if gt:
        title += "  |  green = ground truth"
    show = args.show and use_interactive_backend()
    plot_tracks_3d(tracks, cams, out_dir / "tracks_3d.png", title, pts, fruit, gt, close=not show)
    print(f"\nSaved to {out_dir}  (tracks.csv, tracks.json, summary.txt, tracks_3d.png, "
          f"{len(reproj)} reprojection plots)")
    if show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
