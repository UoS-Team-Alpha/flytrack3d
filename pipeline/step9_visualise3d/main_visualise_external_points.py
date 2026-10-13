"""Show the interactive (rotatable) 3D plot of points or labels that come without their run.

main_visualise_past_3d.py needs the run folder a result was made in. This script needs only
the files themselves and its own config, external_points_config.json (template:
config_examples/external_points_config.example.json), which holds the inputs, the calibration
("cameras") and the few tracking3d / arena_plotting settings it uses; config.json is not read.
Give it one of ("inputs" in the config, or the flags below):

- a csv of 3D positions (--csv): a points.csv of main_points3d.py, a tracks_3d.csv of
  main_tracks3d*.py, or any csv with frame, x, y, z columns (plus n_views and / or track).
  A csv with a track column is drawn as tracks, any other as points coloured by frame.
- YOLO labels (--labels and / or --fg-labels): the detector's predictions and the boxes the
  factor graph inserted (<result>/reprojection/labels/). Each is a folder with one sub-folder
  per camera, named after it, holding one .txt per frame. The two are pooled, each camera's
  frame_offset is added, and the box centres are matched across the cameras and triangulated
  as in main_points3d.py (tracking3d settings of the config), which gives the same plot as
  its points.csv. With --link the points are also linked frame to frame and drawn as tracks.

The fruit, leaves and arena frame are found in the videos, so they are only drawn with
videos: a folder with one video (or folder of frames) per camera, named after it. Their
fits are cached in <output_dir>/external/<videos folder name>/3d/. There is no ground-truth
overlay. The world frame is the one of the calibration, so a csv made with another
calibration will not line up with the cameras.

From the IDE: fill in the config (or the SETTINGS block below) and press Run (or F5).
From the command line, flags override the settings and the config; giving --csv, --labels or
--fg-labels replaces all three of the config's:
    python pipeline/step9_visualise3d/main_visualise_external_points.py                 # everything from the config
    python pipeline/step9_visualise3d/main_visualise_external_points.py --csv path/to/points.csv
    python pipeline/step9_visualise3d/main_visualise_external_points.py --csv path/to/tracks_3d.csv --videos path/to/videos
    python pipeline/step9_visualise3d/main_visualise_external_points.py --labels path/to/predictions --fg-labels path/to/reprojection/labels
    python pipeline/step9_visualise3d/main_visualise_external_points.py --labels path/to/run_folder --link --save plot.png
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]   # the repository root
sys.path.insert(0, str(ROOT))                # so the project packages import when this file is run directly

from scene3d import locate_arena  # noqa: E402
from common.config import load_config  # noqa: E402
from tracking3d.external import (ExternalSession, external_views, find_sources, link_points,  # noqa: E402
                                 load_csv, triangulate_views)
from tracking3d.multiview import tracking_settings, views_summary  # noqa: E402
from visualise.plots3d.scene import plot_points_3d, plot_tracks_3d, use_interactive_backend  # noqa: E402

# ============================================================================ SETTINGS
CONFIG_PATH = ROOT / "external_points_config.json"   # "inputs", "cameras", "tracking3d", "arena_plotting"
# None = the value in the config's "inputs". Setting any of the first three replaces all three.
CSV = None           # a points.csv / tracks_3d.csv
LABELS = None        # folder of predicted YOLO labels, one sub-folder per camera
FG_LABELS = None     # folder of factor-graph YOLO labels (reprojection/labels), one sub-folder per camera
VIDEOS = None        # folder with one video per camera, to draw the fruit, leaves and arena
LINK = None          # labels only: also link the triangulated points into tracks (no factor graph)
ARENA = None         # draw the fake fruit, leaves and arena frame (needs videos; config arena_plotting)
SAVE = None          # also save the plot to this .png
SHOW = None          # open the interactive (rotatable) 3D window
# =====================================================================================


def _path(value) -> Path | None:
    return Path(value).expanduser().resolve() if value else None


def main():
    parser = argparse.ArgumentParser(description="Show the 3D plot of an external points / tracks csv or YOLO labels.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--csv", default=CSV, help="points.csv, tracks_3d.csv or any csv with frame, x, y, z.")
    parser.add_argument("--labels", default=LABELS, help="Predicted YOLO labels: one sub-folder per camera.")
    parser.add_argument("--fg-labels", default=FG_LABELS,
                        help="Factor-graph YOLO labels (reprojection/labels): one sub-folder per camera.")
    parser.add_argument("--videos", default=VIDEOS, help="Folder with one video per camera, for the arena objects.")
    parser.add_argument("--conf", type=float, default=None, help="Override tracking3d.conf_threshold (--labels).")
    parser.add_argument("--link", action=argparse.BooleanOptionalAction, default=LINK,
                        help="Link the points triangulated from the labels into tracks.")
    parser.add_argument("--arena", action=argparse.BooleanOptionalAction, default=ARENA,
                        help="Draw the fake fruit, leaves and arena frame (needs --videos).")
    parser.add_argument("--save", default=SAVE, help="Also save the plot to this .png.")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=SHOW)
    args = parser.parse_args()

    if not Path(args.config).is_file():
        raise SystemExit(f"Config not found: {args.config}\nCopy config_examples/external_points_config.example.json "
                         f"to external_points_config.json and fill it in.")
    config = load_config(args.config)
    inputs = {k: v for k, v in (config.get("inputs") or {}).items() if not k.startswith("_")}
    what = ("csv", "labels", "fg_labels")
    given = {k: getattr(args, k) for k in what}
    if any(given.values()):   # points / labels given here replace the config's, not add to them
        inputs.update(given)
    for k in ("videos", "link", "arena", "save", "show"):
        if getattr(args, k) is not None:
            inputs[k] = getattr(args, k)
    args.link, args.arena, args.show = (bool(inputs.get(k, d)) for k, d in (("link", False), ("arena", True), ("show", True)))
    args.save = inputs.get("save")

    csv_path, labels, fg_labels, videos = (_path(inputs.get(k)) for k in (*what, "videos"))
    if (csv_path is None) == (labels is None and fg_labels is None):
        raise SystemExit("Give either a csv, or labels and / or fg_labels: in 'inputs' of the config, "
                         "or with --csv / --labels / --fg-labels.")
    if csv_path is not None and not csv_path.is_file():
        raise SystemExit(f"csv not found: {csv_path}")
    cam_cfg = config.get("cameras") or {}
    if not cam_cfg.get("views"):
        raise SystemExit("config has no 'cameras' -> 'views'. See config_examples/external_points_config.example.json.")
    s = tracking_settings(config)
    if args.conf is not None:
        s["conf_threshold"] = args.conf

    views = external_views(cam_cfg, find_sources(videos, cam_cfg), labels, fg_labels, s)
    if not views:
        raise SystemExit("No camera to draw: none in cameras.views is calibrated"
                         + ("." if csv_path else " and has a sub-folder in the label folders."))
    cams = [v.camera for v in views]

    if csv_path is not None:
        name = f"{csv_path.parent.name}/{csv_path.name}"
        points, tracks = load_csv(csv_path)
        print(f"\nRead {name}: " + (f"{len(points)} points" if tracks is None else f"{len(tracks)} track(s)"))
        grey = None
    else:
        name = " + ".join(p.name for p in (labels, fg_labels) if p)
        print(f"\nDetections ({name}):\n" + "\n".join(views_summary(views)))
        points = triangulate_views(views, s)
        if not len(points):
            raise SystemExit("\nNo detections could be matched across views. Check the calibration, the cameras' "
                             "frame_offset and max_epipolar_px / max_reprojection_px in tracking3d.")
        print(f"Triangulated {len(points)} points in {len(np.unique(points[:, 0]))} frames")
        tracks, grey = (link_points(points, s), points) if args.link else (None, None)
        if args.link and not tracks:
            raise SystemExit(f"No {s['min_tracklet_frames']}+ frame track could be linked; run without --link.")

    arena = None
    if args.arena and videos is not None:
        cache = Path(config["output_dir"]) / "external" / videos.name
        arena = locate_arena(ExternalSession(SimpleNamespace(path=cache, name=videos.name), views), config)
    elif args.arena:
        print("\nNo videos given: the fruit, leaves and arena frame are not drawn.")

    show = args.show and use_interactive_backend()
    if args.show and not show:
        print("No interactive matplotlib backend (TkAgg / QtAgg) is available to show the plot.")
    save = _path(args.save)
    if tracks is None:
        title = f"{name}  |  {len(points)} points (colour = frame, large = 3 views)"
        plot_points_3d(points, cams, save, title, close=not show, arena=arena)
    else:
        title = f"{name}  |  {len(tracks)} {'linked ' if args.link else ''}tracks (triangle = entry, square = exit)"
        plot_tracks_3d(tracks, cams, save, title, grey, close=not show, arena=arena)
    if save:
        print(f"\nSaved {save}")
    if show:
        print("\nClose the window to finish.")
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
