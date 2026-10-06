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
    tracks.json / summary.txt  per-track entry point, length, speed
    tracks_3d.png            3D + orthographic views with the raw triangulated points
    reprojection/<cam>.png   tracks projected back into each camera (calibration check)
"""

import argparse
from pathlib import Path

import numpy as np

from evaluate2d.config import load_config
from tracking3d.factor_graph import build_tracks
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
N_TRACKS = None      # keep the n best tracks; None = tracking3d.n_tracks (null = every valid track)
GT_OVERLAY = True    # draw the run's ground truth (triangulated + linked, no factor graph) in green
SHOW = True          # open an interactive (rotatable) 3D window after saving
# =====================================================================================


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

    out_dir = session.out_dir("tracks")
    report = write_track_outputs(session, tracks, s, out_dir, f"{len(tracks)} track(s), best supported first")
    print("\n" + report)

    pts = np.array([[f, *p.xyz, len(p.views)] for f, ps in points.items() for p in ps])
    reproj = save_reprojection_plots(tracks, views, out_dir / "reprojection")
    title = f"{session.run.name}  |  {session.label}  |  {len(tracks)} tracks (triangle = entry, square = exit)"
    gt = ground_truth_tracks(session, config["cameras"]) if args.gt_overlay and args.source == "pred" else None
    if gt:
        title += "  |  green = ground truth"
    show = args.show and use_interactive_backend()
    plot_tracks_3d(tracks, cams, out_dir / "tracks_3d.png", title, pts, gt, close=not show)
    print(f"\nSaved to {out_dir}  (tracks.csv, tracks.json, summary.txt, tracks_3d.png, "
          f"{len(reproj)} reprojection plots)")
    if show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
