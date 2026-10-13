"""3D track results as text: the per-track summary, the 3D evaluation against ground truth and
the 2D evaluation with the tracks back-projected (results as computed by tracking3d)."""

from __future__ import annotations

from .detection import fmt, pr_iou_key


def format_summary(session, tracks: list, summaries: list[dict], s: dict, heading: str) -> str:
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


def format_2d(r: dict, data: list, score: float) -> str:
    """data: tracking3d.evaluate.ViewData per view (only its conf is used)."""
    first = r["ALL (pooled)"]["detector"]
    pr_key = pr_iou_key(first)
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
            lines.append(f"  {view if i == 0 else '':<{width}}{name:<16}{res['boxes']:>7}{fmt(at['precision']):>9}"
                         f"{fmt(at['recall']):>9}{fmt(at['f1']):>9}"
                         + "".join(f"{fmt(res['mAP'][m]['all']):>10}" for m in maps))
        lines.append("")
    if score < data[0].conf:
        lines.append(f"  NOTE: backproj_score {score:g} is below the P/R confidence {data[0].conf:g}, so inserted "
                     f"boxes only count towards mAP.")
    return "\n".join(lines)
