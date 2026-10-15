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


CENTRE_ROWS = {   # point set -> its sub-rows: (tag, label)
    "detector": [],
    "+ factor graph": [("inserted", "from factor graph")],   # its detector boxes are the detector row's
    "tracked detections": [],
    "track reprojection": [("observed", "frames with a detection"), ("bridged", "frames bridged")],
}


def _centre_rows(view: str, by_set: dict, width: int, size: bool) -> list[str]:
    """Rows of the centre-error table (size=False) or the size-error table (size=True, box sets only)."""
    rows = []
    for name, res in by_set.items():
        if size and name == "track reprojection":
            continue   # points, not boxes
        for tag, label in [("all", name)] + [(t, "  " + lab) for t, lab in CENTRE_ROWS[name]]:
            st = res.get(tag, {"points": 0, "matched": 0, "precision": None})
            row = f"  {view if not rows else '':<{width}}{label:<26}{st['points']:>7}{st['matched']:>8}"
            if size:
                if st["matched"]:
                    row += (f"{st['iou_mean']:>9.3f}{100 * st['size_err_median']:>15.1f}%{100 * st['size_err_p95']:>8.1f}%"
                            f"{100 * st['size_bias_median']:>+9.1f}%")
            else:
                recall = fmt(st["recall"]) if tag == "all" and st.get("recall") is not None else "    - "
                prec = fmt(st["precision"]) if st["precision"] is not None else "  n/a "
                row += f"{recall:>9}{prec:>11}"
                if st["matched"]:
                    row += "".join(f"{st[k]:>8.2f}" for k in ("rmse", "mean", "median", "p95", "max"))
            rows.append(row)
    return rows


def format_2d(r: dict, data: list, score: float, track_conf: float | None, centres: dict | None = None,
              gate_px: float | None = None) -> str:
    """data: tracking3d.evaluate.ViewData per view (only its conf is used); track_conf: tracking3d.conf_threshold;
    centres: tracking3d.evaluate.centre_errors (matched within gate_px)."""
    first = r["ALL (pooled)"]["detector"]
    pr_keys = list(first["precision_recall"])
    pr_keys.sort(key=lambda k: k != pr_iou_key(first))   # the headline IoU first, the others after it
    width = max(len(n) for n in r) + 2
    conf = data[0].conf
    tags = [k.replace("IoU@", "@") for k in pr_keys]
    lines = ["2D evaluation with the factor-graph tracks projected back into each camera", "",
             f"  detector       : the 2D detector alone (YOLO): its saved predictions, as main_evaluate2d.py scored them",
             f"  + factor graph : plus a box wherever a track has no detection in that view (score {score:g})",
             f"  on tracks only : only the predictions a track used "
             f"({'every saved box' if track_conf is None else f'conf >= {track_conf:g}'}, tracking3d.conf_threshold), "
             f"plus those boxes",
             f"  Compare the rows of a view to see what the factor graph adds to the detector.", ""]

    # ---- 1. detection: P/R/F1 at the headline IoU in full, the other IoUs alongside
    head, rest = pr_keys[0], pr_keys[1:]
    lines += [f"  1. Detection: boxes matched to the ground truth by IoU (COCO matching), at conf >= {conf:g} "
              f"(evaluation.pr_conf_threshold); all sizes",
              f"  {'view':<{width}}{'predictions':<16}{'boxes':>7}{'P' + tags[0]:>9}{'R' + tags[0]:>9}{'F1' + tags[0]:>9}"
              f"{'TP':>7}{'FP':>7}{'FN':>7}" + "".join(f"{'P' + t:>9}{'R' + t:>9}{'F1' + t:>9}" for t in tags[1:])]
    for view, by_variant in r.items():
        for i, (name, res) in enumerate(by_variant.items()):
            at = res["precision_recall"][head]["at_conf"]
            row = (f"  {view if i == 0 else '':<{width}}{name:<16}{res['boxes']:>7}{fmt(at['precision']):>9}"
                   f"{fmt(at['recall']):>9}{fmt(at['f1']):>9}{at['tp']:>7}{at['fp']:>7}{at['fn']:>7}")
            for k in rest:
                o = res["precision_recall"][k]["at_conf"]
                row += f"{fmt(o['precision']):>9}{fmt(o['recall']):>9}{fmt(o['f1']):>9}"
            lines.append(row)
        lines.append("")
    if rest:
        lines += [f"  The {', '.join(tags[1:])} columns are more lenient: the gap to {tags[0]} counts the boxes that sit on",
                  f"  the fly but with a poor size or shape. TP / FP / FN at every IoU are in evaluation.json.", ""]

    if centres:
        # ---- 2. centre error: position only
        n_gt = centres["ALL (pooled)"]["detector"]["gt"]
        lines += [f"  2. Centre error (position): each point set matched one-to-one to the ground-truth box centres of",
                  f"  every frame by distance (Hungarian, within {gate_px:g} px, tracking3d.eval_match_px); no IoU, box size",
                  f"  or ranking, so the rows compare directly:",
                  f"    detector           : the detector's box centres (conf >= {conf:g})",
                  f"    + factor graph     : plus the centres of the inserted boxes (indented: those alone)",
                  f"    tracked detections : the raw detector boxes the tracks were built from",
                  f"    track reprojection : the tracks themselves, projected into every frame: where the track has a",
                  f"                         detection in that view, and where it bridged a gap (the 2D counterpart of",
                  f"                         evaluation_3d.txt, against the labels instead of triangulated ground truth)",
                  f"  'tracked detections' vs 'frames with a detection' = the same flies in the same frames, raw",
                  f"  detector vs the factor graph's smoothed position.",
                  f"  recall = ground-truth centres matched / all ({n_gt} pooled); precision = points matched / points.",
                  f"  Distances (raw image px) are of the matched points only.",
                  f"  {'view':<{width}}{'points':<26}{'n':>7}{'matched':>8}{'recall':>9}{'precision':>11}"
                  f"{'RMSE':>8}{'mean':>8}{'median':>8}{'p95':>8}{'max':>8}"]
        for view, by_set in centres.items():
            lines += _centre_rows(view, by_set, width, size=False) + [""]

        # ---- 3. size error: the boxes matched in 2
        lines += [f"  3. Box size: the boxes matched in 2 against their ground-truth box. Box scale = sqrt(w * h);",
                  f"  size error = |predicted / true scale - 1|; bias = median of (predicted / true - 1), + = too large.",
                  f"  The inserted boxes take their size from the track's own detections in that view, scaled by depth.",
                  f"  {'view':<{width}}{'boxes':<26}{'n':>7}{'matched':>8}{'IoU mean':>9}"
                  f"{'size err median':>16}{'p95':>9}{'bias':>10}"]
        for view, by_set in centres.items():
            lines += _centre_rows(view, by_set, width, size=True) + [""]

    # ---- 4. COCO AP of the detector, laid out as in step 5's summary.txt
    lines += [f"  4. COCO AP of the detector alone (YOLO; as main_evaluate2d.py's summary.txt): 101-pt, "
              f"maxDets={first['settings']['max_dets_per_image']},",
              f"  over every saved box (the whole precision-recall curve). The factor-graph sets have none: their",
              f"  inserted boxes all score {score:g} (backproj_score), so an AP over them would follow that constant,",
              f"  and 'on tracks only' holds just the boxes tracking used, so its curve stops there."]
    areas = list(next(iter(first["average_precision"].values())))
    lines.append(f"  {'view':<{width}}{'':<14}" + "".join(f"{a:>10}" for a in areas))
    for view, by_variant in r.items():
        for i, (ap, by_area) in enumerate(by_variant["detector"]["average_precision"].items()):
            lines.append(f"  {view if i == 0 else '':<{width}}{ap:<14}" + "".join(f"{fmt(by_area[a]):>10}" for a in areas))
        lines.append("")

    if score < conf:
        lines += [f"  NOTE: backproj_score {score:g} is below the P/R confidence {conf:g}, so the inserted "
                  f"boxes are left out of P / R / F1, and the factor-graph rows show only what they drop."]
    return "\n".join(lines).rstrip()
