"""3D results handed over without their run: a csv of 3D points / tracks, or folders of YOLO
labels, plotted with nothing but the calibration in config["cameras"]
(main_visualise_external_points.py).

A csv is read by its header, so it can be a points.csv (main_points3d.py: frame, x, y, z,
n_views, ...), a tracks_3d.csv (main_tracks3d*.py: track, frame, x, y, z, ...) or anything
else with frame, x, y, z columns. Its frames are taken as already aligned across the views.

A label folder holds one sub-folder per camera, whose name contains the camera's name (or
its "match" string), with one YOLO .txt per frame anywhere below it ("class xc yc w h [conf]",
normalised; the frame number is the digits ending the file name). This is the layout of a
run's predictions (<run>/<input>/<model>/predictions/, so a run folder with one model works)
and of the boxes the factor graph inserted (<result>/reprojection/labels/). Several label
folders are pooled per camera and frame, each camera's frame_offset is added, and the box
centres are matched across the views and triangulated exactly as main_points3d.py does.

Videos (or frame folders) are only needed for the arena objects, which are segmented from
them; they are matched to the cameras by name in the same way.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from common.datasets import SPLIT_LISTS, VIDEO_EXTS
from common.linking import link_tracks

from .camera import camera_from_config
from .multiview import View, _frame_no, _read_centres, match_cameras, remove_static_detections
from .past_results import SavedTrack
from .triangulate import MatchSettings, triangulate_session


@dataclass
class ExternalSession:
    """The part of a Session the arena objects need (arena_plotting.locate_arena): the views, and
    a folder to cache the fits in (run.path / "3d")."""
    run: SimpleNamespace   # path, name
    views: list[View]


# ----------------------------------------------------------------------------- csv

def load_csv(path: Path) -> tuple[np.ndarray | None, list[SavedTrack] | None]:
    """(points, None) for a csv of points: (N, 5) frame, x, y, z, n_views (2 if it has no such
    column); (None, tracks) for one with a "track" column."""
    with open(path, newline="") as fh:
        header = [h.strip().lower() for h in next(csv.reader(fh), [])]
    missing = [c for c in ("frame", "x", "y", "z") if c not in header]
    if missing:
        raise SystemExit(f"{path} has no column {missing}; its header is {header}")
    cols = [header.index(c) for c in ("frame", "x", "y", "z")]
    extra = [header.index(c) for c in ("n_views", "track") if c in header]
    rows = np.loadtxt(path, delimiter=",", skiprows=1, usecols=cols + extra, ndmin=2)
    if not len(rows):
        raise SystemExit(f"{path} holds no rows.")
    n_views = rows[:, 4] if "n_views" in header else np.full(len(rows), 2.0)
    if "track" not in header:
        return np.column_stack([rows[:, :4], n_views]), None
    tracks = []
    for tid in np.unique(rows[:, -1]):
        r, nv = rows[rows[:, -1] == tid], n_views[rows[:, -1] == tid]
        order = np.argsort(r[:, 0], kind="stable")
        tracks.append(SavedTrack(int(tid), r[order, 0].astype(int), r[order, 1:4], int(nv.sum())))
    return None, tracks


# ----------------------------------------------------------------------------- views

def _match(root: Path, cam_cfg: dict, what: str) -> dict[str, Path]:
    """camera -> the entry of `root` named after it."""
    if not root.is_dir():
        raise SystemExit(f"{what} folder not found: {root}")
    entries = {p.name: p for p in sorted(root.iterdir())}
    matches, problems = match_cameras(list(entries), cam_cfg)
    if problems:
        raise SystemExit(f"{what} folder {root}: " + "; ".join(problems))
    return {cam: entries[name] for cam, name in matches.items()}


def find_sources(videos: Path | None, cam_cfg: dict) -> dict[str, dict]:
    """camera -> its dataset source ({"video": file} or {"images": folder of frames}) in `videos`."""
    if videos is None:
        return {}
    found = _match(videos, cam_cfg, "Videos")
    sources = {cam: {"images": str(p)} if p.is_dir() else {"video": str(p)}
               for cam, p in found.items() if p.is_dir() or p.suffix.lower() in VIDEO_EXTS}
    if not sources:
        print(f"  No video in {videos} is named after a camera ({', '.join(cam_cfg['views'])}).")
    return sources


def _video_size(source: dict) -> tuple[int, int] | None:
    if not source.get("video"):
        return None
    cap = cv2.VideoCapture(source["video"])
    size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return size if all(size) else None


def _read_label_folder(folder: Path, size: tuple[int, int], frame_offset: int) -> dict[int, np.ndarray]:
    """{aligned frame: (N, 3) x, y, conf in raw pixels} of every label file below a camera's folder."""
    out, seen = {}, {}
    for p in sorted(folder.rglob("*.txt")):
        fno = _frame_no(p.stem)
        if fno is None or p.name in SPLIT_LISTS:
            continue
        try:
            c = _read_centres(p, *size, with_conf=True)
        except (ValueError, IndexError):
            continue   # some other text file
        if fno in seen:
            raise SystemExit(f"{folder} holds two label files for frame {fno}:\n  {seen[fno]}\n  {p}\n"
                             f"Point at a folder with one set of labels per camera (one model, one sequence).")
        seen[fno] = p
        if len(c):
            out[fno + frame_offset] = c
    return out


def external_views(cam_cfg: dict, sources: dict[str, dict], pred_labels: Path | None = None,
                   fg_labels: Path | None = None, settings: dict | None = None) -> list[View]:
    """One View per camera with a filled-in calibration, in config order: its detections from the
    label folders (none if there are no labels, e.g. for a csv) and its video, if any.
    The prediction labels are filtered as open_session does (conf_threshold, static clutter);
    the factor-graph labels are boxes on tracks and are all kept."""
    s = settings or {}
    folders = [(_match(root, cam_cfg, "Label"), is_pred)
               for root, is_pred in ((pred_labels, True), (fg_labels, False)) if root is not None]
    labelled = set().union(*(f for f, _ in folders)) if folders else None
    views, removed = [], {}
    for name, cfg in cam_cfg["views"].items():
        if name.startswith("_") or (labelled is not None and name not in labelled):
            continue
        source = sources.get(name, {})
        try:
            camera = camera_from_config(name, cfg, _video_size(source), float(cam_cfg.get("t_scale", 1.0)))
        except ValueError as e:
            print(f"  {name}: left out ({e})")
            continue
        input_name = Path(source.get("video") or source.get("images") or name).stem
        view = View(camera, input_name, camera.image_size or (0, 0), {}, {}, source=source)
        for found, is_pred in folders:
            if name not in found:
                continue
            if not camera.image_size:
                raise SystemExit(f"Unknown image size for '{name}': give its video, or image_size in cameras.views.")
            raw = _read_label_folder(found[name], camera.image_size, camera.frame_offset)
            if is_pred:
                if s.get("conf_threshold") is not None:
                    raw = {f: c[c[:, 2] >= s["conf_threshold"]] for f, c in raw.items()}
                    raw = {f: c for f, c in raw.items() if len(c)}
                part = View(camera, input_name, view.size, raw,
                            {f: np.hstack([camera.undistort(c[:, :2]), c[:, 2:3]]) for f, c in raw.items()})
                if s.get("static_min_fraction"):
                    removed[name] = remove_static_detections(part, s["static_radius_px"], s["static_min_fraction"])
                raw = part.raw
            for f, c in raw.items():
                view.raw[f] = np.vstack([view.raw[f], c]) if f in view.raw else c
        view.dets = {f: np.hstack([camera.undistort(c[:, :2]), c[:, 2:3]]) for f, c in view.raw.items()}
        views.append(view)
    if removed:
        print(f"Static clutter removed from the predictions (same spot within {s['static_radius_px']} px in >= "
              f"{s['static_min_fraction']:.0%} of frames): " + ", ".join(f"{n} {k}" for n, k in removed.items()))
    return views


# ----------------------------------------------------------------------------- labels -> 3D

def triangulate_views(views: list[View], s: dict) -> np.ndarray:
    """(N, 5) frame, x, y, z, n_views: the views' detections matched and triangulated per frame."""
    if len(views) < 2:
        raise SystemExit("Need labels for at least 2 calibrated cameras to triangulate.")
    points = triangulate_session([v.camera for v in views], [v.dets for v in views],
                                 MatchSettings(s["max_epipolar_px"], s["max_reprojection_px"]))
    return np.array([[f, *p.xyz, len(p.views)] for f, ps in points.items() for p in ps]).reshape(-1, 5)


def link_points(points: np.ndarray, s: dict) -> list[SavedTrack]:
    """Triangulated points linked frame to frame into tracks (Hungarian on 3D distance, as the
    ground-truth overlay is built): no factor graph, so a track is only as smooth as its points."""
    seq = [(int(f), points[points[:, 0] == f, 1:4]) for f in np.unique(points[:, 0])]
    linked = [t for t in link_tracks(seq, s["link_max_distance"], int(s["link_max_frame_gap"]))
              if len(t) >= int(s["min_tracklet_frames"])]
    linked.sort(key=len, reverse=True)
    return [SavedTrack(i, t[:, 0].astype(int), t[:, 1:4], len(t)) for i, t in enumerate(linked, 1)]
