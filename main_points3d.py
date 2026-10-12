"""Triangulate the 2D detections of a previous batch evaluation run into 3D points.

Every camera view in the chosen run is treated as a view of the same recording
session (cameras and their calibration come from config["cameras"]). Detections
are matched across views per frame with the Hungarian method on epipolar distance;
anything seen in fewer than two views is dropped. No temporal tracking - for
that, see main_tracks3d.py.

From the IDE: edit the SETTINGS block below and press Run (or F5).
From the command line, flags override the settings:
    python main_points3d.py                     # pick a run from a list
    python main_points3d.py --run run_20261006-113511_640_960inf_960window
    python main_points3d.py --source gt         # triangulate the ground-truth labels instead
    python main_points3d.py --no-show           # only save files

Outputs go to <run>/3d/<model or ground_truth>/points/.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from arena_plotting import locate_arena
from evaluate2d.config import load_config
from tracking3d.multiview import ground_truth_tracks, open_session
from tracking3d.triangulate import MatchSettings, camera_baseline_report, triangulate_session
from tracking3d.visualise import plot_points_3d, use_interactive_backend

HERE = Path(__file__).resolve().parent

# ============================================================================ SETTINGS
CONFIG_PATH = HERE / "config.json"   # needs "cameras" (calibration) and optionally "tracking3d"
RUN = None           # run folder name or path; None = choose in the terminal
MODEL = None         # model in that run; None = the only one, or choose in the terminal
SOURCE = "pred"      # "pred" = the run's predictions, "gt" = its ground-truth labels
GT_OVERLAY = True    # draw the run's ground truth (triangulated + linked, no factor graph) in green
ARENA = True         # draw the fake fruit and leaves, found by HSV in each view (config arena_plotting)
SHOW = True          # open an interactive (rotatable) 3D window after saving
# =====================================================================================


def main():
    parser = argparse.ArgumentParser(description="Triangulate multi-view 2D detections into 3D points.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--run", default=RUN)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--source", choices=["pred", "gt"], default=SOURCE)
    parser.add_argument("--conf", type=float, default=None, help="Override tracking3d.conf_threshold.")
    parser.add_argument("--gt-overlay", action=argparse.BooleanOptionalAction, default=GT_OVERLAY,
                        help="Overlay ground-truth 3D tracks in green (ignored with --source gt).")
    parser.add_argument("--arena", action=argparse.BooleanOptionalAction, default=ARENA,
                        help="Draw the fake fruit and its leaves (found by HSV in every view).")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=SHOW)
    args = parser.parse_args()

    config = load_config(args.config)
    session = open_session(config, args.run, args.model, args.source, args.conf)
    views, s = session.views, session.settings
    cams = [v.camera for v in views]
    print("\nCamera rig:\n" + "\n".join(camera_baseline_report(cams)))

    points = triangulate_session(cams, [v.dets for v in views],
                                 MatchSettings(s["max_epipolar_px"], s["max_reprojection_px"]))
    flat = [(f, p) for f, pts in points.items() for p in pts]
    if not flat:
        raise SystemExit("\nNo detections could be matched across views. Check the calibration "
                         "(cameras.views) and max_epipolar_px / max_reprojection_px in tracking3d.")

    by_views = Counter(len(p.views) for _, p in flat)
    reproj = np.array([p.reproj_px for _, p in flat])
    n_det = sum(len(d) for v in views for d in v.dets.values())
    used = sum(len(p.views) for _, p in flat)
    print(f"\nTriangulated {len(flat)} points in {len(points)} frames "
          f"({', '.join(f'{n} views: {c}' for n, c in sorted(by_views.items()))})")
    print(f"  detections used : {used}/{n_det} ({100 * used / max(n_det, 1):.1f}%)")
    print(f"  reprojection px : median {np.median(reproj):.2f}, 90th pct {np.percentile(reproj, 90):.2f}")

    out_dir = session.out_dir("points")
    with open(out_dir / "points.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "x", "y", "z", "n_views", "reprojection_px", "conf"]
                   + [f"det_{v.name}" for v in views])
        for f, p in flat:
            w.writerow([f, *p.xyz.round(4), len(p.views), round(p.reproj_px, 3), round(p.conf, 4)]
                       + [p.views.get(i, "") for i in range(len(views))])

    arr = np.array([[f, *p.xyz, len(p.views)] for f, p in flat])
    title = (f"{session.run.name}  |  {session.label}  |  {len(flat)} triangulated points "
             f"(colour = frame, large = 3 views)")
    gt = ground_truth_tracks(session, config["cameras"]) if args.gt_overlay and args.source == "pred" else None
    if gt:
        title += "  |  green = ground truth"
    arena = locate_arena(session, config) if args.arena else None
    show = args.show and use_interactive_backend()
    plot_points_3d(arr, cams, out_dir / "points_3d.png", title, gt, close=not show,
                   arena=arena)
    print(f"\nSaved {out_dir / 'points.csv'}\n      {out_dir / 'points_3d.png'}")
    if show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
