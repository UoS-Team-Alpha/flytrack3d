"""2D detector metrics as text: the per-model report, the side-by-side comparison and the
per-input table of a batch run (results as computed by detection.metrics)."""

from __future__ import annotations


def fmt(v: float) -> str:
    return "  n/a " if v is None or v < 0 else f"{v:.4f}"


def ap_tag(name: str) -> str:
    """Column header for an AP metric: "AP@0.5" / "mAP@0.5" -> "@0.5"."""
    return name[name.index("@"):] if "@" in name else name


def ap_values(results: dict) -> list[float]:
    """The all-sizes value of each AP metric, in config order. By position rather than by name, as
    one input of a batch can be single-class ("AP@...") while the pooled result is not ("mAP@...")."""
    return [by_area["all"] for by_area in results["average_precision"].values()]


def pr_iou_key(results: dict) -> str:
    """IoU threshold used for headline P/R/F1: 0.5 if evaluated, else the last one."""
    pr = results["precision_recall"]
    return "IoU@0.5" if "IoU@0.5" in pr else list(pr)[-1]


def format_model_report(name: str, info: dict, results: dict, class_names: dict) -> str:
    cfg = info["model_config"]
    timing = info["timing"]
    lines = [
        "=" * 78,
        f" MODEL: {name}",
        "=" * 78,
        f"  type/model_type : {cfg.get('type', 'sahi')} / {cfg.get('model_type', '-')}",
        f"  weights         : {cfg.get('model_path')}",
        f"  conf threshold  : {cfg.get('confidence_threshold')}",
    ]
    if cfg.get("type", "sahi") == "sahi":
        lines.append(
            f"  slicing         : {cfg.get('slice_width')}x{cfg.get('slice_height')}, overlap "
            f"{cfg.get('overlap_width_ratio')}/{cfg.get('overlap_height_ratio')}, "
            f"post {cfg.get('postprocess_type')}/{cfg.get('postprocess_match_metric')}"
            f"@{cfg.get('postprocess_match_threshold')}, full-frame pass={cfg.get('perform_standard_pred')}"
        )
    lines += [
        f"  predictions     : {info['prediction_source']}  (cache key {info['cache_key']})",
        f"  speed           : {timing.get('ms_per_frame')} ms/frame, {timing.get('fps')} FPS "
        f"over {timing.get('frames')} frames",
        f"  total dets      : {results['settings']['predictions']}",
        "",
        f"  {'Mean average precision' if results['settings']['gt_classes'] > 1 else 'Average precision'} "
        f"(COCO 101-pt, maxDets={results['settings']['max_dets_per_image']})",
    ]
    areas = list(next(iter(results["average_precision"].values())).keys())
    lines.append("  " + f"{'':<14}" + "".join(f"{a:>10}" for a in areas))
    for map_name, by_area in results["average_precision"].items():
        lines.append("  " + f"{map_name:<14}" + "".join(f"{fmt(by_area[a]):>10}" for a in areas))

    lines += ["", "  Precision / recall (all sizes)"]
    lines.append("  " + f"{'':<22}{'conf':>7}{'P':>9}{'R':>9}{'F1':>9}{'TP':>8}{'FP':>8}{'FN':>8}")
    for iou_name, d in results["precision_recall"].items():
        for label, key in (("at conf", "at_conf"), ("best F1", "best_f1")):
            r = d[key]
            lines.append(
                "  " + f"{iou_name + ' ' + label:<22}{r['conf']:>7.3f}{r['precision']:>9.4f}"
                f"{r['recall']:>9.4f}{r['f1']:>9.4f}{r['tp']:>8}{r['fp']:>8}{r['fn']:>8}"
            )

    if len(results["per_class"]) > 1:
        lines += ["", "  Per-class AP (all sizes)"]
        ap_names = list(next(iter(results["per_class"].values())))
        lines.append("  " + f"{'class':<20}" + "".join(f"{m:>14}" for m in ap_names))
        for c, v in results["per_class"].items():
            label = class_names.get(int(c), c)
            lines.append("  " + f"{str(label):<20}" + "".join(f"{fmt(v[m]['all']):>14}" for m in ap_names))
    return "\n".join(lines)


def format_comparison(all_results: dict) -> str:
    """One-line-per-model summary for quick side-by-side comparison."""
    if len(all_results) < 2:
        return ""
    first = next(iter(all_results.values()))["precision_recall"]
    pr_key = "IoU@0.5" if "IoU@0.5" in first else list(first)[-1]
    tag = pr_key.replace("IoU@", "@")
    cols = [(f"P{tag}", lambda r: r["precision_recall"][pr_key]["at_conf"]["precision"]),
            (f"R{tag}", lambda r: r["precision_recall"][pr_key]["at_conf"]["recall"])]
    for i, m in enumerate(next(iter(all_results.values()))["average_precision"]):
        cols.append((ap_tag(m), lambda r, i=i: ap_values(r)[i]))
    lines = ["=" * 78, " COMPARISON (all sizes)", "=" * 78,
             "  " + f"{'model':<24}" + "".join(f"{c:>10}" for c, _ in cols)]
    for name, r in all_results.items():
        lines.append("  " + f"{name:<24}" + "".join(f"{fmt(f(r)):>10}" for _, f in cols))
    return "\n".join(lines)


def format_batch_table(item_results: dict[str, dict], pooled: dict, item_sizes: dict[str, tuple]) -> str:
    """Per-input breakdown for each model, with the pooled result as the last row."""
    lines = ["=" * 78, " PER-INPUT RESULTS (all sizes; P/R/F1 at each model's conf)", "=" * 78]
    width = max(len(n) for n in [*item_results, "ALL (pooled)"]) + 2
    for model, pooled_res in pooled.items():
        pr_key = pr_iou_key(pooled_res)
        tag = pr_key.replace("IoU@", "@")
        map_names = list(pooled_res["average_precision"])
        lines += ["", f"  {model}",
                  "  " + f"{'input':<{width}}{'frames':>7}{'GT':>7}" + f"{'P' + tag:>9}{'R' + tag:>9}"
                  f"{'F1' + tag:>9}" + "".join(f"{ap_tag(m):>10}" for m in map_names)]
        rows = [(n, r[model], item_sizes[n]) for n, r in item_results.items() if model in r]
        rows.append(("ALL (pooled)", pooled_res, item_sizes["ALL (pooled)"]))
        for i, (name, r, (frames, gt)) in enumerate(rows):
            if i == len(rows) - 1:
                lines.append("  " + "-" * (width + 32 + 10 * len(map_names)))
            at = r["precision_recall"][pr_key]["at_conf"]
            lines.append("  " + f"{name:<{width}}{frames:>7}{gt:>7}{fmt(at['precision']):>9}"
                         f"{fmt(at['recall']):>9}{fmt(at['f1']):>9}"
                         + "".join(f"{fmt(v):>10}" for v in ap_values(r)))
    return "\n".join(lines)


def headline(name: str, info: dict, results: dict) -> str:
    """One-line result for the terminal during batch runs."""
    pr_key = pr_iou_key(results)
    at = results["precision_recall"][pr_key]["at_conf"]
    tag = pr_key.replace("IoU@", "@")
    maps = "  ".join(f"{m} {fmt(v['all'])}" for m, v in results["average_precision"].items())
    return (f"    {name:<28} P{tag} {fmt(at['precision'])}  R{tag} {fmt(at['recall'])}  "
            f"F1 {fmt(at['f1'])}  {maps}  [{info['timing'].get('ms_per_frame')} ms/frame, "
            f"{'cached' if info['prediction_source'] != 'inference' else 'inference'}]")
