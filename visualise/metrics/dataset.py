"""Dataset statistics as text: per split, the videos it holds with their image and instance
counts, and the sizes of its boxes (statistics as computed by
dataset_tools.image_tools.analysis.dataset_stats)."""

from __future__ import annotations

import numpy as np

from dataset_tools.image_tools.analysis.dataset_stats import (SplitStats, percentiles, shared_videos,
                                                              size_distribution)

WIDTH = 100


def _share(part: int, whole: int) -> str:
    return f"{100 * part / whole:5.1f}%" if whole else "   n/a"


def _class_breakdown(instances, class_names: dict) -> str:
    return ", ".join(f"{class_names.get(c, f'class {c}')}: {n}" for c, n in sorted(instances.items())) or "-"


def _stat_row(label: str, values: np.ndarray, decimals: int = 1) -> str:
    p = percentiles(values)
    if p is None:
        return f"  {label:<22}" + f"{'n/a':>9}" * 6
    return f"  {label:<22}" + "".join(f"{p[k]:>9.{decimals}f}" for k in ("min", "p5", "median", "mean", "p95", "max"))


def format_split_report(s: SplitStats, class_names: dict) -> str:
    instances = s.instances
    total = sum(instances.values())
    sizes = ", ".join(f"{w}x{h} ({n})" for (w, h), n in s.image_sizes.most_common(5))
    if len(s.image_sizes) > 5:
        sizes += f", ... ({len(s.image_sizes)} sizes)"
    lines = [
        "=" * WIDTH,
        f" SPLIT: {s.name.upper()}",
        "=" * WIDTH,
        f"  images             : {s.image_dir}",
        f"  labels             : {s.label_dir}",
        f"  videos             : {len(s.videos)}",
        f"  images             : {s.images}",
        f"    with objects     : {s.images - s.background}  ({_share(s.images - s.background, s.images).strip()})",
        f"    background       : {s.background}  ({_share(s.background, s.images).strip()})",
        f"  instances          : {total}  ({_class_breakdown(instances, class_names)})",
        f"  image sizes        : {sizes or 'n/a'}",
    ]
    problems = [(s.missing_labels, "images have no label file (counted as background)"),
                (s.orphan_labels, "label files have no image"),
                (s.unreadable_images, "images could not be read (their boxes are left out of the size statistics)"),
                (s.out_of_bounds, "boxes reach outside their image")]
    lines += [f"  WARNING            : {n} {what}" for n, what in problems if n]

    lines += ["", "  Videos (most images first)",
              "  " + f"{'#':>3}{'images':>9}{'share':>8}{'backgr.':>9}{'instances':>11}{'inst/img':>10}{'box px':>8}"
              + "  video prefix"]
    ranked = sorted(s.videos.items(), key=lambda kv: -kv[1].images)
    for i, (video, v) in enumerate(ranked, 1):
        n = sum(v.instances.values())
        box = s.box_sizes(video)
        box_px = f"{np.median(box):>8.1f}" if len(box) else f"{'n/a':>8}"
        row = (f"{i:>3}{v.images:>9}{_share(v.images, s.images):>8}{v.background:>9}{n:>11}"
               f"{n / v.images:>10.2f}{box_px}  {video}")
        if len(instances) > 1:
            row += f"  [{_class_breakdown(v.instances, class_names)}]"
        lines.append("  " + row)

    w, h = s.boxes[:, 3], s.boxes[:, 4]
    lines += ["", "  Distributions (box size = sqrt(width x height))",
              "  " + f"{'':<22}" + "".join(f"{c:>9}" for c in ("min", "5%", "median", "mean", "95%", "max")),
              _stat_row("instances per image", s.boxes_per_image),
              _stat_row("box width (px)", w),
              _stat_row("box height (px)", h),
              _stat_row("box size (px)", s.box_sizes()),
              _stat_row("aspect ratio (w/h)", w / h if len(w) else w, decimals=2)]

    lines += ["", "  Boxes by size (px)"]
    for name, n in size_distribution(s.box_sizes()).items():
        lines.append(f"    {name:<8}{n:>9}  {_share(n, len(s.boxes))}")
    return "\n".join(lines)


def format_dataset_report(dataset_name: str, stats: list[SplitStats], class_names: dict) -> str:
    """The whole report: a totals table over the splits, then each split in detail."""
    images = sum(s.images for s in stats)
    instances = sum(sum(s.instances.values()) for s in stats)
    lines = ["=" * WIDTH, f" DATASET: {dataset_name}", "=" * WIDTH,
             "  " + f"{'split':<12}{'videos':>8}{'images':>10}{'share':>8}{'backgr.':>10}{'instances':>11}{'share':>8}"
             f"{'inst/img':>10}{'box px':>8}"]
    for s in stats:
        n = sum(s.instances.values())
        box = s.box_sizes()
        box_px = f"{np.median(box):>8.1f}" if len(box) else f"{'n/a':>8}"
        lines.append("  " + f"{s.name:<12}{len(s.videos):>8}{s.images:>10}{_share(s.images, images):>8}"
                     f"{s.background:>10}{n:>11}{_share(n, instances):>8}"
                     f"{n / s.images if s.images else 0:>10.2f}{box_px}")
    lines.append("  " + f"{'total':<12}{len({v for s in stats for v in s.videos}):>8}{images:>10}{'':>8}"
                 f"{sum(s.background for s in stats):>10}{instances:>11}")
    lines.append("  (backgr. = images without an object; box px = median box size, sqrt(width x height))")

    shared = shared_videos(stats)
    if shared:
        lines += ["", f"  WARNING: {len(shared)} videos are in more than one split, so their frames leak between them:"]
        lines += [f"    {' + '.join(splits):<16}{video}" for video, splits in shared.items()]
    elif len(stats) > 1:
        lines += ["", "  No video is in more than one split."]
    return "\n".join(lines + [""] + [format_split_report(s, class_names) + "\n" for s in stats])
