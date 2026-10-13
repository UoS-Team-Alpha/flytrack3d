"""Charts of what a YOLO dataset holds (statistics as computed by
dataset_tools.stats).

    contents  pie charts: the images and instances of each split, and of each video within a split
    boxes     per split: where the boxes sit in the frame, their width against their height, and
              a histogram of their size

A pie shows its largest MAX_SLICES - 1 videos and pools the rest as "Other"; every slice is
named with its count in the legend under the pie. A video keeps its colour in both pies of a
split, and across the splits too when the dataset has few enough videos for that.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from dataset_tools.stats import SplitStats

# Categorical colours, used in this fixed order; grey is kept for "Other" / background.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
OTHER = "#b5b4ae"
SURFACE = "#ffffff"
INK, INK_MUTED = "#0b0b0b", "#52514e"
SPLIT_SLOTS = {"train": 0, "val": 1, "valid": 1, "validation": 1, "test": 2}
MAX_SLICES = len(SERIES)
LABEL_CHARS = 44          # longer video names keep their end, which is what tells them apart
HEATMAP = "Blues"


def _short(name: str) -> str:
    return name if len(name) <= LABEL_CHARS else "..." + name[-(LABEL_CHARS - 3):]


def _pie(fig, cell, title: str, slices: list[tuple[str, float, str]]) -> None:
    """A pie in a grid cell with its legend underneath. slices: (label, value, colour)."""
    inner = cell.subgridspec(2, 1, height_ratios=[3, 2], hspace=0.02)
    ax, legend_ax = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
    legend_ax.axis("off")
    ax.set_title(title, fontsize=11, color=INK)
    slices = [s for s in slices if s[1] > 0]
    total = sum(v for _, v, _ in slices)
    if not total:
        ax.axis("off")
        ax.text(0.5, 0.5, "none", ha="center", va="center", color=INK_MUTED, transform=ax.transAxes)
        return
    wedges, _, _ = ax.pie([v for _, v, _ in slices], colors=[c for _, _, c in slices], startangle=90,
                          counterclock=False, wedgeprops={"edgecolor": SURFACE, "linewidth": 2},
                          autopct=lambda p: f"{p:.0f}%" if p >= 4 else "", pctdistance=1.14,
                          textprops={"color": INK, "fontsize": 9})
    legend_ax.legend(wedges, [f"{_short(label)}  -  {v:,.0f} ({100 * v / total:.1f}%)" for label, v, _ in slices],
                     loc="upper center", fontsize=8, frameon=False, labelcolor=INK_MUTED, handlelength=1.2)


def _split_colours(stats: list[SplitStats]) -> dict[str, str]:
    """train, val and test keep their colour whichever splits are analysed; other names take the free ones."""
    free = [c for i, c in enumerate(SERIES) if i not in {SPLIT_SLOTS.get(s.name) for s in stats}]
    return {s.name: SERIES[SPLIT_SLOTS[s.name]] if s.name in SPLIT_SLOTS else free.pop(0) for s in stats}


def _video_colours(stats: list[SplitStats]) -> dict[str, dict[str, str]]:
    """{split: {video: colour}} for the videos that get a slice of their own; the rest are pooled as "Other"."""
    totals: dict[str, int] = {}
    for s in stats:
        for video, v in s.videos.items():
            totals[video] = totals.get(video, 0) + v.images
    if len(totals) <= MAX_SLICES:   # few enough videos: one colour per video over the whole dataset
        ranked = sorted(totals, key=lambda video: -totals[video])
        colours = {video: SERIES[i] for i, video in enumerate(ranked)}
        return {s.name: {video: colours[video] for video in s.videos} for s in stats}
    out = {}
    for s in stats:
        ranked = sorted(s.videos, key=lambda video: -s.videos[video].images)
        keep = MAX_SLICES if len(ranked) <= MAX_SLICES else MAX_SLICES - 1
        out[s.name] = {video: SERIES[i] for i, video in enumerate(ranked[:keep])}
    return out


def _video_slices(s: SplitStats, colours: dict[str, str], value) -> list[tuple[str, float, str]]:
    slices = [(video, value(s.videos[video]), colour) for video, colour in colours.items()]
    pooled = [value(v) for video, v in s.videos.items() if video not in colours]
    if pooled:
        slices.append((f"Other ({len(pooled)} videos)", sum(pooled), OTHER))
    return slices


def plot_contents(stats: list[SplitStats], class_names: dict, dataset_name: str = ""):
    """Pie charts of the dataset: its splits on the top row, then each split's videos by images and by instances."""
    split_colours = _split_colours(stats)
    instances = sum((s.instances for s in stats), Counter())
    images, background = sum(s.images for s in stats), sum(s.background for s in stats)
    overview = []
    if len(stats) > 1:
        overview += [("Images per split", [(s.name, s.images, split_colours[s.name]) for s in stats]),
                     ("Instances per split", [(s.name, sum(s.instances.values()), split_colours[s.name])
                                              for s in stats])]
    overview.append(("Images with / without objects",
                     [("with objects", images - background, SERIES[0]), ("background", background, OTHER)]))
    if len(instances) > 1:
        ranked = sorted(instances.items(), key=lambda kv: -kv[1])
        slices = [(str(class_names.get(c, f"class {c}")), n, SERIES[i]) for i, (c, n) in enumerate(ranked[:MAX_SLICES - 1])]
        if len(ranked) > MAX_SLICES - 1:
            slices.append((f"Other ({len(ranked) - MAX_SLICES + 1} classes)", sum(n for _, n in ranked[MAX_SLICES - 1:]), OTHER))
        overview.append(("Instances per class", slices))

    cols = max(len(overview), len(stats))
    fig = plt.figure(figsize=(5.4 * cols, 15.5), facecolor=SURFACE)
    grid = GridSpec(3, cols, figure=fig, left=0.02, right=0.98, top=0.93, bottom=0.01, hspace=0.12, wspace=0.05)
    fig.suptitle(f"Dataset contents{': ' + dataset_name if dataset_name else ''}", fontsize=14, color=INK)
    for col, (title, slices) in enumerate(overview):
        _pie(fig, grid[0, col], title, slices)
    video_colours = _video_colours(stats)
    for col, s in enumerate(stats):
        n = sum(s.instances.values())
        _pie(fig, grid[1, col], f"{s.name}: images per video ({s.images:,} images, {len(s.videos)} videos)",
             _video_slices(s, video_colours[s.name], lambda v: v.images))
        _pie(fig, grid[2, col], f"{s.name}: instances per video ({n:,} instances)",
             _video_slices(s, video_colours[s.name], lambda v: sum(v.instances.values())))
    return fig


def _empty(ax, text: str = "no boxes") -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", color=INK_MUTED, transform=ax.transAxes)


def plot_boxes(stats: list[SplitStats], dataset_name: str = ""):
    """Per split: a heatmap of the box centres, a heatmap of box width against height, and a box size histogram."""
    fig, axes = plt.subplots(len(stats), 3, figsize=(19, 5.6 * len(stats)), squeeze=False, facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.4, 1, 1.2]}, layout="constrained")
    fig.suptitle(f"Bounding boxes{': ' + dataset_name if dataset_name else ''}", fontsize=14, color=INK)
    for (ax_pos, ax_dim, ax_size), s in zip(axes, stats):
        if not len(s.boxes):
            for ax in (ax_pos, ax_dim, ax_size):
                _empty(ax, f"{s.name}: no boxes")
            continue
        x_c, y_c, w, h = s.boxes[:, 1], s.boxes[:, 2], s.boxes[:, 3], s.boxes[:, 4]

        # Where the boxes sit: in px if every image has the same size, else in fractions of the image
        (img_w, img_h), unit = (next(iter(s.image_sizes)), "px") if len(s.image_sizes) == 1 else ((1, 1), "fraction of the image")
        (frame_w, frame_h), _ = s.image_sizes.most_common(1)[0]
        bins_y = max(8, round(48 * frame_h / frame_w))
        *_, mesh = ax_pos.hist2d(x_c * img_w, y_c * img_h, bins=(48, bins_y), range=((0, img_w), (0, img_h)), cmap=HEATMAP)
        fig.colorbar(mesh, ax=ax_pos, label="boxes", shrink=0.8)
        ax_pos.set_ylim(img_h, 0)   # image coordinates: origin top-left
        ax_pos.set_aspect(frame_h / frame_w * img_w / img_h)
        sizes = f"{frame_w}x{frame_h}" if len(s.image_sizes) == 1 else f"{len(s.image_sizes)} image sizes"
        ax_pos.set_title(f"{s.name}: box centres ({len(s.boxes):,} boxes, {sizes})", fontsize=11, color=INK)
        ax_pos.set_xlabel(f"x ({unit})")
        ax_pos.set_ylabel(f"y ({unit})")

        # Width against height, zoomed to where nearly all the boxes are
        limit = float(max(np.percentile(w, 99), np.percentile(h, 99))) * 1.05
        inside = int(((w <= limit) & (h <= limit)).sum())
        *_, mesh = ax_dim.hist2d(w, h, bins=60, range=((0, limit), (0, limit)), cmap=HEATMAP)
        fig.colorbar(mesh, ax=ax_dim, label="boxes", shrink=0.8)
        ax_dim.plot([0, limit], [0, limit], color=INK_MUTED, lw=0.8, ls=":")   # square boxes
        ax_dim.set_aspect("equal", adjustable="box")
        ax_dim.set_title(f"{s.name}: box width x height ({len(w) - inside} boxes beyond the axes)", fontsize=11, color=INK)
        ax_dim.set_xlabel("width (px)")
        ax_dim.set_ylabel("height (px)")

        size = np.sqrt(w * h)
        median = float(np.median(size))
        ax_size.hist(size, bins=60, range=(0, limit), color=SERIES[0], edgecolor=SURFACE, linewidth=0.5)
        ax_size.axvline(median, color=INK, lw=1, ls="--")
        ax_size.annotate(f"median {median:.1f} px", (median, 1), xycoords=("data", "axes fraction"), xytext=(5, -12),
                         textcoords="offset points", fontsize=9, color=INK)
        ax_size.set_title(f"{s.name}: box size, sqrt(width x height)", fontsize=11, color=INK)
        ax_size.set_xlabel("box size (px)")
        ax_size.set_ylabel("boxes")
        ax_size.spines[["top", "right"]].set_visible(False)
        ax_size.grid(axis="y", color="#e4e3df", lw=0.6)
        ax_size.set_axisbelow(True)
    return fig


def show_dataset_charts(stats: list[SplitStats], class_names: dict, dataset_name: str = "",
                        out_dir: str | Path | None = None, show: bool = True) -> list[Path]:
    """Draw both figures, save them to out_dir (if given) as contents.png and boxes.png, and open them (show)."""
    figures = {"contents": plot_contents(stats, class_names, dataset_name), "boxes": plot_boxes(stats, dataset_name)}
    saved = []
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            saved.append(Path(out_dir) / f"{name}.png")
            fig.savefig(saved[-1], dpi=110)
    if show:
        print("Showing the charts (close the windows to continue)...")
        plt.show()
    for fig in figures.values():
        plt.close(fig)
    return saved
