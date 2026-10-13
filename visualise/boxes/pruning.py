"""Figures for checking what dataset pruning finds and drops (the frames are chosen by
dataset_tools.prune.review): sample frames of a bbox size rank, and dropped frames next to the kept frame
each one matched. Close a window to carry on in the terminal.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt

from .yolo_dataset import plot_yolo_boxes, zoom_to_box


def show_box_samples(samples: list[tuple], names: list[str], title: str, grid: bool = True,
                     window_title: str | None = None) -> None:
    """Show images with their boxes drawn. samples: (image path, boxes) pairs; names: what to call
    each one. grid = all in one numbered grid with the names listed underneath, else one window
    each. window_title labels the window itself (e.g. the rank), as well as the figure title."""
    if grid:
        cols = math.ceil(math.sqrt(len(samples)))
        img_rows = math.ceil(len(samples) / cols)
        legend_height = 0.25 * len(samples) + 0.3

        fig = plt.figure(figsize=(4 * cols, 3.5 * img_rows + legend_height))
        gs = fig.add_gridspec(img_rows + 1, cols, height_ratios=[3.5] * img_rows + [legend_height])

        for i, (img_path, boxes) in enumerate(samples):
            ax = fig.add_subplot(gs[i // cols, i % cols])
            plot_yolo_boxes(ax, img_path, boxes, number=i + 1)

        # Numbered list of image names under the grid for cross-referencing
        legend_ax = fig.add_subplot(gs[img_rows, :])
        legend_ax.axis('off')
        legend_text = "\n".join(f"{i}: {name}" for i, name in enumerate(names, 1))
        legend_ax.text(0, 1, legend_text, va='top', ha='left', family='monospace', fontsize=9)

        fig.suptitle(title)
        if window_title:
            fig.canvas.manager.set_window_title(window_title)
        fig.tight_layout()
        plt.show()
    else:
        for i, ((img_path, boxes), name) in enumerate(zip(samples, names), 1):
            fig, ax = plt.subplots(figsize=(10, 8))
            plot_yolo_boxes(ax, img_path, boxes)
            ax.set_xlabel(name, fontsize=10)
            ax.set_title(f"{title}\nImage {i}/{len(samples)}")
            if window_title:
                fig.canvas.manager.set_window_title(f"{window_title} ({i}/{len(samples)})")
            fig.tight_layout()
            plt.show()


def show_dropped_pairs(pairs: list[dict], title: str) -> None:
    """One row per dropped frame: dropped (left) next to the kept frame it matched (right), both
    zoomed in on the box that was compared. pairs: dicts with img_path, bboxes, box, rank, ssim
    and name for the dropped frame, and matched_img_path (None if not found), matched_bboxes,
    matched_box and matched_name for the kept one."""
    fig, axes = plt.subplots(len(pairs), 2, figsize=(8, 3.8 * len(pairs)), squeeze=False)
    for (ax_drop, ax_kept), pair in zip(axes, pairs):
        plot_yolo_boxes(ax_drop, pair["img_path"], pair["bboxes"])
        zoom_to_box(ax_drop, pair["img_path"], pair["box"])
        ax_drop.set_title(f"DROPPED | rank {pair['rank']} | SSIM {pair['ssim']}\n{pair['name']}", fontsize=8)
        if pair["matched_img_path"]:
            plot_yolo_boxes(ax_kept, pair["matched_img_path"], pair["matched_bboxes"])
            zoom_to_box(ax_kept, pair["matched_img_path"], pair["matched_box"])
            ax_kept.set_title(f"KEPT (matched)\n{pair['matched_name']}", fontsize=8)
        else:
            ax_kept.axis('off')
            ax_kept.set_title("Matched frame not found", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
