"""Bar charts of the evaluation results, saved as PNGs next to the text reports.

    save_detection_charts  step 5: AP per IoU, AP@0.5 by object size, P / R / F1 at the operating
                           point, and (batch runs) AP@0.5 per input - one colour per model
    save_2d_charts         step 8, evaluation_2d.txt: P / R / F1 of detector vs + factor graph vs
                           on tracks only, centre error and box size error of each point set
    save_3d_charts         step 8, evaluation_3d.txt: 3D error by number of views, recall /
                           precision, and how much of each ground-truth track the tracks cover

Every bar is labelled with its value (the charts are a picture of the text report, which holds the
exact numbers). Colours are the categorical slots of visualise.charts.dataset, in fixed order.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .dataset import INK, INK_MUTED, SERIES, SURFACE

GRID = "#e4e3df"
BAR_GAP = 2   # px of surface between touching bars


def _style(ax, title: str, ylabel: str = "") -> None:
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_ylabel(ylabel, fontsize=9, color=INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_facecolor(SURFACE)


def _bars(ax, groups: list[str], series: dict[str, list[float | None]], fmt: str = "{:.2f}",
          ylim: tuple | None = None, colours: list[str] | None = None) -> None:
    """Grouped bars: one group per entry of groups, one bar per series (in fixed colour order).
    None = no value (no bar, "n/a" written in its place). A legend for 2+ series."""
    colours = colours or list(SERIES)
    n = len(series)
    width = min(0.8 / max(n, 1), 0.28)
    x = np.arange(len(groups))
    top = max([v for vals in series.values() for v in vals if v is not None] + [0])
    for i, (name, vals) in enumerate(series.items()):
        pos = x + (i - (n - 1) / 2) * width
        h = [0 if v is None else v for v in vals]
        bars = ax.bar(pos, h, width, color=colours[i % len(colours)], edgecolor=SURFACE, linewidth=BAR_GAP,
                      label=name, zorder=2)
        for b, v in zip(bars, vals):
            ax.annotate("n/a" if v is None else fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                        fontsize=7, color=INK_MUTED if v is None else INK)
    ax.set_xticks(x, groups)
    ax.set_ylim(*(ylim or (0, top * 1.18 if top else 1)))
    if n > 1:
        ax.legend(fontsize=8, frameon=False, labelcolor=INK_MUTED, loc="upper left", bbox_to_anchor=(1, 1))


def _save(fig, path: Path) -> Path:
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def _distinct(names: list[str], limit: int = 28) -> list[str]:
    """Short labels that keep what tells the names apart: their common start and end are cut."""
    if len(names) < 2:
        return [n[-limit:] for n in names]
    pre = len(os.path.commonprefix(names))
    suf = len(os.path.commonprefix([n[::-1] for n in names]))
    core = [n[pre:len(n) - suf] or n for n in names]
    if min(map(len, core)) < 3:   # the cut ate the difference: keep the whole names
        core = names
    return [c if len(c) <= limit else c[:limit - 3] + "..." for c in core]


def _valid(v: float | None) -> float | None:
    return None if v is None or v < 0 else float(v)


# ----------------------------------------------------------------------------- step 5

def save_detection_charts(path: Path, all_results: dict[str, dict], per_input: dict[str, dict] | None = None,
                          title: str = "") -> Path:
    """all_results: {model: detection.metrics result}; per_input: {input: {model: result}} (batch)."""
    models = list(all_results)
    first = all_results[models[0]]
    ap_names = list(first["average_precision"])
    head_ap = next((n for n in ap_names if n.endswith("@0.5")), ap_names[0])
    areas = [a for a in next(iter(first["average_precision"].values())) if a != "all"]
    pr_keys = list(first["precision_recall"])

    panels = 3 + (per_input is not None and len(per_input) > 1)
    fig, axes = plt.subplots(panels, 1, figsize=(9, 3.0 * panels), constrained_layout=True)
    _bars(axes[0], ap_names, {m: [_valid(r["average_precision"][n]["all"]) for n in ap_names]
                              for m, r in all_results.items()}, ylim=(0, 1.08))
    _style(axes[0], "AP over every saved box (all sizes)", "AP")
    _bars(axes[1], areas, {m: [_valid(r["average_precision"][head_ap][a]) for a in areas]
                           for m, r in all_results.items()}, ylim=(0, 1.08))
    _style(axes[1], f"{head_ap} by object size", "AP")
    conf = first["settings"]["pr_conf"]
    groups = [f"{s} {k.replace('IoU', '')}" for k in pr_keys for s in ("P", "R", "F1")]
    _bars(axes[2], groups, {m: [r["precision_recall"][k]["at_conf"][s] for k in pr_keys
                                for s in ("precision", "recall", "f1")] for m, r in all_results.items()},
          ylim=(0, 1.08))
    _style(axes[2], f"Precision / recall / F1 at conf >= {conf:g}" if conf is not None else
           "Precision / recall / F1 (every saved box)", "")
    if panels == 4:
        names = list(per_input)
        short = _distinct(names)
        _bars(axes[3], short, {m: [_valid(per_input[n][m]["average_precision"][head_ap]["all"])
                                   if m in per_input[n] else None for n in names] for m in models},
              ylim=(0, 1.08))
        _style(axes[3], f"{head_ap} per input (all sizes)", "AP")
    if title:
        fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    return _save(fig, path)


# ----------------------------------------------------------------------------- step 8, 2D

def save_2d_charts(path: Path, r: dict, centres: dict | None) -> Path:
    """r: tracking3d.evaluate.evaluate_2d; centres: tracking3d.evaluate.centre_errors (pooled rows charted)."""
    pooled = r["ALL (pooled)"]
    variants = list(pooled)
    pr_keys = list(pooled[variants[0]]["precision_recall"])
    views = [v for v in r if v != "ALL (pooled)"]
    head = "IoU@0.5" if "IoU@0.5" in pr_keys else pr_keys[-1]

    panels = 2 + 2 * bool(centres)
    fig, axes = plt.subplots(panels, 1, figsize=(9, 3.0 * panels), constrained_layout=True)
    groups = [f"{s} {k.replace('IoU', '')}" for k in pr_keys for s in ("P", "R", "F1")]
    conf = pooled[variants[0]]["settings"]["pr_conf"]
    _bars(axes[0], groups, {v: [pooled[v]["precision_recall"][k]["at_conf"][s] for k in pr_keys
                                for s in ("precision", "recall", "f1")] for v in variants}, ylim=(0, 1.1))
    _style(axes[0], f"Detection, all views pooled (conf >= {conf:g})", "")
    _bars(axes[1], views, {v: [r[view][v]["precision_recall"][head]["at_conf"]["f1"] for view in views]
                           for v in variants}, ylim=(0, 1.1))
    _style(axes[1], f"F1{head.replace('IoU', '')} per view", "F1")

    if centres:
        c = centres["ALL (pooled)"]
        sets = [("detector", c["detector"].get("all", {})),
                ("factor graph boxes", c["+ factor graph"].get("inserted", {})),
                ("tracked detections", c["tracked detections"].get("all", {})),
                ("track, detected frames", c["track reprojection"].get("observed", {})),
                ("track, bridged frames", c["track reprojection"].get("bridged", {}))]
        _bars(axes[2], [s for s, _ in sets], {stat: [st.get(key) for _, st in sets]
                                              for stat, key in (("median", "median"), ("p95", "p95"),
                                                                ("RMSE", "rmse"))})
        _style(axes[2], "Centre error of matched points, all views pooled (lower is better)", "px")
        boxes = sets[:3]
        _bars(axes[3], [s for s, _ in boxes], {stat: [None if st.get(key) is None else 100 * st[key]
                                                      for _, st in boxes]
                                               for stat, key in (("median error", "size_err_median"),
                                                                 ("p95 error", "size_err_p95"))}, fmt="{:.1f}%")
        for i, (_, st) in enumerate(boxes):   # the bias, under each group's bars
            if st.get("size_bias_median") is not None:
                axes[3].annotate(f"bias {100 * st['size_bias_median']:+.1f}%", (i, 0), xytext=(0, -24),
                                 textcoords="offset points", ha="center", fontsize=7, color=INK_MUTED)
        _style(axes[3], "Box size error, sqrt(w*h) vs ground truth (lower is better; bias: + = too large)", "%")
    return _save(fig, path)


# ----------------------------------------------------------------------------- step 8, 3D

def save_3d_charts(path: Path, r: dict) -> Path:
    """r: tracking3d.evaluate.evaluate_3d."""
    rows = {"all": r["error"], **r["error_by_views"]}
    fig, axes = plt.subplots(3, 1, figsize=(9, 9.5), constrained_layout=True,
                             gridspec_kw={"height_ratios": [3, 2, 3]})
    groups = [f"{k}\n(n={e['n']})" for k, e in rows.items()]
    _bars(axes[0], groups, {label: [e.get(key) if e["n"] else None for e in rows.values()]
                            for label, key in (("median", "median"), ("p95", "p95"), ("RMSE", "rmse"))})
    _style(axes[0], "3D error of matched track points, by cameras with a detection (lower is better)",
           "world units (mm)")
    _bars(axes[1], ["recall", "precision"], {"": [r["recall"], r["precision"]]}, ylim=(0, 1.1))
    _style(axes[1], f"Track points matched to the ground truth (within {r['match_distance']:g})", "")
    gts = r["per_gt_track"]
    _bars(axes[2], [str(g["id"]) for g in gts], {"": [100 * g["matched"] / g["points"] if g["points"] else 0
                                                      for g in gts]}, fmt="{:.0f}%", ylim=(0, 112))
    _style(axes[2], "Share of each ground-truth track covered by a track (x: ground-truth track id)", "%")
    return _save(fig, path)
