"""Track summaries and output files shared by main_tracks3d.py and main_tracks3d_guided.py."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from .factor_graph import Track3D


def format_summary(session, tracks: list[Track3D], summaries: list[dict], s: dict, heading: str) -> str:
    lines = [f"Session   : {session.run.name}",
             f"Detections: {session.label}",
             f"Views     : " + ", ".join(f"{v.name} <- {v.input_name}" for v in session.views),
             f"Settings  : " + ", ".join(f"{k}={v}" for k, v in s.items() if not k.startswith("_")), "",
             heading, ""]
    head = f"  {'id':>3}{'frames':>13}{'len':>6}{'dets':>6}{'3-view':>8}{'2-view':>8}{'1-view':>8}" \
           f"{'reproj':>8}{'sigma':>8}{'path':>10}  {'entry xyz':<26}{'+/-':>7}{'2+ views @':>11}"
    lines.append(head)
    for t in summaries:
        fv = t["frames_by_views"]
        row = (f"  {t['id']:>3}{t['start_frame']:>6}-{t['end_frame']:<6}{t['frames']:>6}{t['detections']:>6}"
               f"{fv.get(3, 0):>8}{fv.get(2, 0):>8}{fv.get(1, 0):>8}"
               f"{t['reprojection_px_median'] or 0:>8.2f}{t['sigma_median'] or float('nan'):>8.2f}"
               f"{t['path_length']:>10.1f}  {'(' + ', '.join(f'{v:.1f}' for v in t['entry_xyz']) + ')':<26}"
               f"{t['entry_sigma'] or float('nan'):>7.1f}{t['first_multiview_frame']:>11}")
        lines.append(row)
    lines += ["", "  frames by views: frames where the track was detected in 3 / 2 / 1 cameras "
                  "(others are bridged by the motion prior)",
              "  reproj = median px error of its detections; sigma = median 1-sigma 3D uncertainty",
              "  entry +/- = 1-sigma at the entry point (large while only one camera sees it: depth then",
              "  comes from the motion prior alone); 2+ views @ = first frame seen by two or more cameras"]
    return "\n".join(lines)


def write_track_outputs(session, tracks: list[Track3D], s: dict, out_dir: Path, heading: str,
                        extra_lines: list[str] = (), extra_json: dict | None = None,
                        params: dict | None = None) -> str:
    """summary.txt, tracks.json, tracks_3d.csv and params.json (the settings this run used, plus
    `params`). Returns the summary text."""
    summaries = [t.summary() for t in tracks]
    report = format_summary(session, tracks, summaries, s, heading)
    if extra_lines:
        report += "\n\n" + "\n".join(extra_lines)
    (out_dir / "summary.txt").write_text(report + "\n")
    (out_dir / "tracks.json").write_text(json.dumps({"run": session.run.name, "detections": session.label,
                                                     "settings": s, "tracks": summaries, **(extra_json or {})},
                                                    indent=2, default=str))
    (out_dir / "params.json").write_text(json.dumps({
        "created": datetime.now().isoformat(timespec="seconds"), "run": session.run.name,
        "detections": session.label, "views": {v.name: v.input_name for v in session.views},
        "frame_offsets": {v.name: v.camera.frame_offset for v in session.views},
        **(params or {}), "tracking3d": s}, indent=2, default=str))
    with open(out_dir / "tracks_3d.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["track", "frame", "x", "y", "z", "sigma", "n_views"])
        for t in tracks:
            for f, p, sg, nv in zip(t.frames, t.xyz, t.sigma, t.views):
                w.writerow([t.id, int(f), *p.round(4), round(float(sg), 4), int(nv)])
    return report
