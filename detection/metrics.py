"""COCO-style detection metrics, implemented in numpy.

Faithful re-implementation of pycocotools' bbox evaluation (greedy score-ordered
matching, ignore handling for out-of-range object sizes, 101-point interpolated
AP) extended to arbitrary IoU thresholds so AP@0.2 can be reported alongside
the standard AP@0.5 and AP@0.5:0.95. The metrics are named "AP@..." when the
ground truth has one class and "mAP@..." (mean over classes) when it has more.

Inputs are plain arrays so any detector/dataset can be evaluated:
    gts   : {image_key: (N, 5) array  class, x1, y1, x2, y2}
    preds : {image_key: (M, 6) array  x1, y1, x2, y2, score, class}
"""

from __future__ import annotations

import numpy as np

COCO_AREA_RANGES = {
    "all": [0.0, 1e10],
    "small": [0.0, 32.0 ** 2],
    "medium": [32.0 ** 2, 96.0 ** 2],
    "large": [96.0 ** 2, 1e10],
}
REC_THRS = np.linspace(0.0, 1.0, 101)
COCO_IOUS = np.round(np.linspace(0.5, 0.95, 10), 2)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between (D,4) and (G,4) xyxy boxes -> (D,G)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def _box_area(b: np.ndarray) -> np.ndarray:
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])


def _match(ious: np.ndarray, gt_ignore: np.ndarray, iou_thrs: np.ndarray):
    """pycocotools evaluateImg matching. Dets must be score-sorted; gts sorted
    so non-ignored come first. Returns dt_matched (T,D), dt_ignore (T,D)."""
    T, (D, G) = len(iou_thrs), ious.shape
    dt_m = np.zeros((T, D), dtype=bool)
    dt_ig = np.zeros((T, D), dtype=bool)
    gt_m = np.zeros((T, G), dtype=bool)
    for ti, t in enumerate(iou_thrs):
        for d in range(D):
            best, m = min(t, 1 - 1e-10), -1
            for g in range(G):
                if gt_m[ti, g]:
                    continue
                if m > -1 and not gt_ignore[m] and gt_ignore[g]:
                    break  # only ignored gts left, and we already have a real match
                if ious[d, g] < best:
                    continue
                best, m = ious[d, g], g
            if m == -1:
                continue
            dt_ig[ti, d] = gt_ignore[m]
            dt_m[ti, d] = True
            gt_m[ti, m] = True
    return dt_m, dt_ig


def _average_precision(tp: np.ndarray, fp: np.ndarray, n_gt: int) -> float:
    """101-point interpolated AP from score-ordered tp/fp flags."""
    if n_gt == 0:
        return -1.0
    if len(tp) == 0:
        return 0.0
    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    rc = tp_c / n_gt
    pr = tp_c / np.maximum(tp_c + fp_c, np.spacing(1))
    pr = np.maximum.accumulate(pr[::-1])[::-1]  # monotone precision envelope
    inds = np.searchsorted(rc, REC_THRS, side="left")
    q = np.zeros(len(REC_THRS))
    valid = inds < len(pr)
    q[valid] = pr[inds[valid]]
    return float(q.mean())


def _pr_at_conf(scores, tp, fp, n_gt, conf) -> dict:
    keep = scores >= conf
    n_tp, n_fp = int(tp[keep].sum()), int(fp[keep].sum())
    p = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0
    r = n_tp / n_gt if n_gt else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"conf": float(conf), "precision": p, "recall": r, "f1": f1,
            "tp": n_tp, "fp": n_fp, "fn": int(n_gt - n_tp)}


def _best_f1(scores, tp, fp, n_gt) -> dict:
    if len(scores) == 0 or n_gt == 0:
        return _pr_at_conf(scores, tp, fp, n_gt, 1.0)
    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    p = tp_c / np.maximum(tp_c + fp_c, 1e-12)
    r = tp_c / n_gt
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    # Only evaluate at the last detection of each distinct score (a threshold keeps all ties).
    last_of_score = np.r_[scores[1:] != scores[:-1], True]
    i = np.flatnonzero(last_of_score)[np.argmax(f1[last_of_score])]
    return _pr_at_conf(scores, tp, fp, n_gt, scores[i])


def match_boxes(gt: np.ndarray, dt: np.ndarray, iou_thr: float, max_dets: int = 100) -> list[tuple[int, int, float]]:
    """The true positives of one image as (dt row, gt row, IoU), matched as evaluate_detections
    does (per class, score order, each detection takes the unmatched ground truth it overlaps
    most at IoU >= iou_thr; max_dets per class). gt: (N, 5) class + xyxy; dt: (M, 6) xyxy, score, class."""
    pairs = []
    for c in np.unique(np.r_[gt[:, 0], dt[:, 5]]):
        g_rows, d_rows = np.flatnonzero(gt[:, 0] == c), np.flatnonzero(dt[:, 5] == c)
        d_rows = d_rows[np.argsort(-dt[d_rows, 4], kind="mergesort")][:max_dets]
        ious = box_iou(dt[d_rows, :4], gt[g_rows, 1:5])
        taken = np.zeros(len(g_rows), bool)
        for i, d in enumerate(d_rows):
            cand = np.where(taken, -1.0, ious[i]) if len(g_rows) else np.zeros(0)
            if len(cand) and cand.max() >= min(iou_thr, 1 - 1e-10):
                j = int(cand.argmax())
                taken[j] = True
                pairs.append((int(d), int(g_rows[j]), float(ious[i, j])))
    return pairs


def ap_name(name: str, n_classes: int) -> str:
    """"mAP@0.5" / "AP@0.5" as "AP@0.5" for one class, "mAP@0.5" for more (a mean over classes)."""
    base = name[1:] if name.startswith("mAP") else name
    return "m" + base if n_classes > 1 and base.startswith("AP") else base


def evaluate_detections(
    gts: dict[str, np.ndarray],
    preds: dict[str, np.ndarray],
    map_iou_thresholds: dict[str, list[float]] | None = None,
    pr_iou_thresholds: list[float] = (0.2, 0.5),
    area_ranges: dict[str, list[float]] | None = None,
    max_dets: int = 100,
    pr_conf: float | None = None,
) -> dict:
    """Compute AP per (IoU setting x size range) and P/R/F1 per IoU threshold.

    map_iou_thresholds: named IoU sets, each averaged into one AP, e.g.
        {"mAP@0.2": [0.2], "mAP@0.5": [0.5], "mAP@0.5:0.95": [0.5, ..., 0.95]}
        The names are reported as "AP@..." when the ground truth has one class, "mAP@..." otherwise.
    pr_conf: confidence for precision/recall (None = keep every saved prediction).
    """
    if map_iou_thresholds is None:
        map_iou_thresholds = {"mAP@0.2": [0.2], "mAP@0.5": [0.5], "mAP@0.5:0.95": COCO_IOUS.tolist()}
    area_ranges = area_ranges or COCO_AREA_RANGES

    all_thrs = sorted({round(t, 4) for ts in map_iou_thresholds.values() for t in ts}
                      | {round(t, 4) for t in pr_iou_thresholds})
    iou_thrs = np.array(all_thrs)
    t_index = {t: i for i, t in enumerate(all_thrs)}

    keys = sorted(set(gts) | set(preds))
    empty_gt, empty_dt = np.zeros((0, 5)), np.zeros((0, 6))
    classes = sorted({int(c) for k in keys for c in gts.get(k, empty_gt)[:, 0]}
                     | {int(c) for k in keys for c in preds.get(k, empty_dt)[:, 5]})
    # Only classes with ground truth enter the mean (the others' AP is undefined).
    gt_classes = {int(c) for k in keys for c in gts.get(k, empty_gt)[:, 0]}

    # acc[area][cls] -> lists of per-image (scores, dt_m, dt_ig) and non-ignored gt count
    acc = {a: {c: {"scores": [], "m": [], "ig": [], "n_gt": 0} for c in classes} for a in area_ranges}

    for k in keys:
        g_all, d_all = gts.get(k, empty_gt), preds.get(k, empty_dt)
        for c in classes:
            g = g_all[g_all[:, 0] == c, 1:5]
            d = d_all[d_all[:, 5] == c]
            if len(g) == 0 and len(d) == 0:
                continue
            d = d[np.argsort(-d[:, 4], kind="mergesort")][:max_dets]
            ious_full = box_iou(d[:, :4], g)
            g_area, d_area = _box_area(g), _box_area(d[:, :4])
            for a, (lo, hi) in area_ranges.items():
                g_ig = (g_area < lo) | (g_area > hi)
                order = np.argsort(g_ig, kind="mergesort")  # non-ignored first
                dt_m, dt_ig = _match(ious_full[:, order], g_ig[order], iou_thrs)
                d_out = (d_area < lo) | (d_area > hi)
                dt_ig |= ~dt_m & d_out[None, :]  # unmatched dets outside the size range don't count
                slot = acc[a][c]
                slot["scores"].append(d[:, 4])
                slot["m"].append(dt_m)
                slot["ig"].append(dt_ig)
                slot["n_gt"] += int((~g_ig).sum())

    # Flatten each (area, class) into score-sorted tp/fp arrays per IoU threshold.
    curves: dict = {}
    for a in area_ranges:
        for c in classes:
            s = acc[a][c]
            T = len(iou_thrs)
            scores = np.concatenate(s["scores"]) if s["scores"] else np.zeros(0)
            m = np.concatenate(s["m"], axis=1) if s["m"] else np.zeros((T, 0), bool)
            ig = np.concatenate(s["ig"], axis=1) if s["ig"] else np.zeros((T, 0), bool)
            order = np.argsort(-scores, kind="mergesort")
            scores, m, ig = scores[order], m[:, order], ig[:, order]
            curves[a, c] = (scores, m & ~ig, ~m & ~ig, ig, s["n_gt"])

    def mean_valid(vals):
        vals = [v for v in vals if v >= 0]
        return float(np.mean(vals)) if vals else -1.0

    # ---- AP table: AP name -> area -> value (mean over classes), plus per-class breakdown
    ap_table, per_class = {}, {c: {} for c in classes}
    for config_name, thrs in map_iou_thresholds.items():
        name = ap_name(config_name, len(gt_classes))
        ap_table[name] = {}
        for a in area_ranges:
            class_aps = []
            for c in classes:
                scores, tp, fp, ig, n_gt = curves[a, c]
                aps = []
                for t in thrs:
                    ti = t_index[round(t, 4)]
                    keep = ~ig[ti]
                    aps.append(_average_precision(tp[ti][keep], fp[ti][keep], n_gt))
                ap = mean_valid(aps)
                class_aps.append(ap)
                per_class[c].setdefault(ap_name(config_name, 1), {})[a] = ap  # one class: AP
            ap_table[name][a] = mean_valid(class_aps)

    # ---- precision / recall / F1 (all sizes, summed over classes)
    pr = {}
    for t in pr_iou_thresholds:
        ti = t_index[round(t, 4)]
        parts = [curves["all", c] for c in classes]
        if parts:
            keep = [~p[3][ti] for p in parts]
            scores = np.concatenate([p[0][k] for p, k in zip(parts, keep)])
            tp = np.concatenate([p[1][ti][k] for p, k in zip(parts, keep)])
            fp = np.concatenate([p[2][ti][k] for p, k in zip(parts, keep)])
            n_gt = sum(p[4] for p in parts)
        else:
            scores, tp, fp, n_gt = np.zeros(0), np.zeros(0, bool), np.zeros(0, bool), 0
        order = np.argsort(-scores, kind="mergesort")
        scores, tp, fp = scores[order], tp[order], fp[order]
        conf = pr_conf if pr_conf is not None else (scores.min() if len(scores) else 0.0)
        pr[f"IoU@{t:g}"] = {
            "at_conf": _pr_at_conf(scores, tp, fp, n_gt, conf),
            "best_f1": _best_f1(scores, tp, fp, n_gt),
        }

    return {
        "average_precision": ap_table,
        "precision_recall": pr,
        "per_class": {str(c): v for c, v in per_class.items()},
        "settings": {
            "map_iou_thresholds": map_iou_thresholds,
            "gt_classes": len(gt_classes),
            "pr_iou_thresholds": list(pr_iou_thresholds),
            "area_ranges": area_ranges,
            "max_dets_per_image": max_dets,
            "pr_conf": pr_conf,
            "images": len(keys),
            "predictions": int(sum(len(v) for v in preds.values())),
        },
    }
