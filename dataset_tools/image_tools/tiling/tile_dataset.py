"""Tile a full-frame YOLO dataset into small square crops, to train a detector for sliced (SAHI) inference.

Each image gives:
- Object tiles: a tile at a random position that contains the image's first
  object. Every other object it holds is labelled too. An object the first
  tile misses (or only shows a sliver of) gets a tile of its own, and so on
  until every object is in a tile.
- Empty tiles: for every image without labels, and for a fraction of the others
  (empty_fraction), a background tile with no object in it. Their positions are
  spread evenly over a 3x3 grid of the frame, so the background is sampled from
  the whole field of view rather than from wherever the objects happen to be.

An object cut by a tile's edge keeps its (clipped) box if at least
min_visible_fraction of it is inside the tile, or min_visible_area px^2 is.

The output is a YOLO dataset: images/<split>/<image>_crop_<n>.jpg and
<image>_empty_b<grid cell>.jpg with their labels in labels/<split>/, and a
data.yaml when the input has one. Images must be at least crop_size in both
width and height; pad or resize smaller ones first.
"""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from common.yolo_format import list_images, read_yolo_boxes, split_dirs


@dataclass
class TileSettings:
    crop_size: int = 640                 # tile width and height, px
    empty_fraction: float = 0.1          # fraction of the images that also give an empty background tile
    empty_tries: int = 10                # attempts at finding a tile with no object in it, per image
    min_visible_fraction: float = 0.25   # a clipped box is kept if this fraction of it is inside the tile ...
    min_visible_area: float = 12         # ... or this many px^2 of it are


def yolo_to_abs(box, img_w: int, img_h: int) -> list:
    """Normalised YOLO [cls, x_c, y_c, w, h] -> [cls, x_min, y_min, x_max, y_max] in pixels."""
    cls, x_c, y_c, w, h = box
    return [int(cls), float((x_c - w / 2) * img_w), float((y_c - h / 2) * img_h),
            float((x_c + w / 2) * img_w), float((y_c + h / 2) * img_h)]


def abs_to_yolo_crop(abs_box, crop_x1: int, crop_y1: int, crop_size: int) -> list:
    """Pixel box in the full frame -> YOLO box normalised to the tile whose top-left corner is (crop_x1, crop_y1)."""
    cls, x_min, y_min, x_max, y_max = abs_box
    x_min, x_max, y_min, y_max = x_min - crop_x1, x_max - crop_x1, y_min - crop_y1, y_max - crop_y1
    w, h = (x_max - x_min) / crop_size, (y_max - y_min) / crop_size
    x_c, y_c = (x_min + (x_max - x_min) / 2) / crop_size, (y_min + (y_max - y_min) / 2) / crop_size
    return [int(cls), round(x_c, 6), round(y_c, 6), round(w, 6), round(h, 6)]


def randomized_clamped_crop(yolo_box, img_w: int, img_h: int, crop_size: int, rng: random.Random) -> tuple:
    """A tile (x1, y1, x2, y2) at a random position that holds the whole box and stays inside the image."""
    _, x_min, y_min, x_max, y_max = yolo_to_abs(yolo_box, img_w, img_h)
    min_x1, max_x1 = max(0, x_max - crop_size), min(x_min, img_w - crop_size)
    min_y1, max_y1 = max(0, y_max - crop_size), min(y_min, img_h - crop_size)
    # An object larger than the tile cannot fit: start the tile at the object's corner.
    if min_x1 > max_x1:
        min_x1 = max_x1 = max(0, min(int(x_min), img_w - crop_size))
    if min_y1 > max_y1:
        min_y1 = max_y1 = max(0, min(int(y_min), img_h - crop_size))

    x1 = max(0, min(int(rng.uniform(min_x1, max_x1)), img_w - crop_size))
    y1 = max(0, min(int(rng.uniform(min_y1, max_y1)), img_h - crop_size))
    return x1, y1, x1 + crop_size, y1 + crop_size


def stratified_empty_crop(img_w: int, img_h: int, bins: list[int], crop_size: int, rng: random.Random) -> tuple:
    """A tile (x1, y1, x2, y2, cell) at a random position in the 3x3 grid cell that has the fewest tiles so far.

    The cell's count in bins is not raised here: the caller does that once it
    accepts the tile, so rejected attempts do not skew the counts.
    """
    cell = int(np.argmin(bins))
    row, col = divmod(cell, 3)
    zone_w, zone_h = (img_w - crop_size) / 3, (img_h - crop_size) / 3
    x1 = rng.randint(int(col * zone_w), int((col + 1) * zone_w))
    y1 = rng.randint(int(row * zone_h), int((row + 1) * zone_h))
    return x1, y1, x1 + crop_size, y1 + crop_size, cell


def overlaps(abs_box, window) -> bool:
    _, x_min, y_min, x_max, y_max = abs_box
    x1, y1, x2, y2 = window
    return x_max > x1 and x_min < x2 and y_max > y1 and y_min < y2


def labels_in_crop(abs_boxes: list, window, s: TileSettings) -> dict[int, list]:
    """{box index: YOLO box in the tile} for the boxes the tile shows enough of, clipped to the tile."""
    x1, y1, x2, y2 = window
    labels = {}
    for idx, box in enumerate(abs_boxes):
        if not overlaps(box, window):
            continue
        cls, x_min, y_min, x_max, y_max = box
        clipped = [cls, max(x_min, x1), max(y_min, y1), min(x_max, x2), min(y_max, y2)]
        area = max(1.0, (x_max - x_min) * (y_max - y_min))
        visible = max(0.0, (clipped[3] - clipped[1]) * (clipped[4] - clipped[2]))
        if visible > 0 and (visible / area >= s.min_visible_fraction or visible >= s.min_visible_area):
            labels[idx] = abs_to_yolo_crop(clipped, x1, y1, s.crop_size)
    return labels


def tile_split(image_paths: list[Path], label_dir: Path, out_image_dir: Path, out_label_dir: Path,
               s: TileSettings, rng: random.Random, desc: str = "images") -> dict:
    """Tile the images of one split. Returns its counts: tiles, instances, empty tiles saved / skipped and per cell."""
    import cv2   # imported here: only the tiling itself needs OpenCV

    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)
    # Every empty_every-th image also gives an empty tile, e.g. every 10th for empty_fraction 0.1.
    empty_every = max(1, int(round(1.0 / s.empty_fraction))) if s.empty_fraction and s.empty_fraction > 0 else None
    bins = [0] * 9
    stats = {"tiles": 0, "instances": 0, "empty_saved": 0, "empty_skipped": 0}

    def save_crop(image, window, name: str, labels: list) -> None:
        x1, y1, x2, y2 = window
        cv2.imwrite(str(out_image_dir / f"{name}.jpg"), image[y1:y2, x1:x2])
        (out_label_dir / f"{name}.txt").write_text("".join(f"{l[0]} {l[1]} {l[2]} {l[3]} {l[4]}\n" for l in labels))
        stats["tiles"] += 1
        stats["instances"] += len(labels)

    count = 0   # images read so far
    for image_path in tqdm(image_paths, desc=desc, unit="img"):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        count += 1
        img_h, img_w = image.shape[:2]
        if img_h < s.crop_size or img_w < s.crop_size:
            raise SystemExit(f"Image '{image_path}' ({img_w}x{img_h}) is smaller than the crop size {s.crop_size}.")
        yolo_boxes = read_yolo_boxes(label_dir / f"{image_path.stem}.txt")
        abs_boxes = [yolo_to_abs(box, img_w, img_h) for box in yolo_boxes]

        # --- Empty background tile: for an image without labels, and for every empty_every-th image.
        if (empty_every and count % empty_every == 0) or not yolo_boxes:
            for _ in range(s.empty_tries):
                *window, cell = stratified_empty_crop(img_w, img_h, bins, s.crop_size, rng)
                if not any(overlaps(box, window) for box in abs_boxes):
                    bins[cell] += 1
                    save_crop(image, window, f"{image_path.stem}_empty_b{cell}", [])
                    stats["empty_saved"] += 1
                    break
            else:
                # No object-free tile found: skip, and let a later image take this turn.
                stats["empty_skipped"] += 1
            if not yolo_boxes:
                continue

        # --- Object tiles: one around the first object, then one for each object no tile shows yet.
        covered = set()

        def crop_around(idx: int, number: int) -> bool:
            """Save a tile at a random position around box idx, unless it cuts that box down to a sliver."""
            window = randomized_clamped_crop(yolo_boxes[idx], img_w, img_h, s.crop_size, rng)
            labels = labels_in_crop(abs_boxes, window, s)
            if idx not in labels:
                return False
            save_crop(image, window, f"{image_path.stem}_crop_{number}", list(labels.values()))
            covered.update(labels)
            return True

        crop_around(0, 0)
        n_crops = 1
        for idx in range(len(yolo_boxes)):
            # A tile that clips its own object to a sliver is retried once at a fresh random position.
            if idx not in covered and (crop_around(idx, n_crops) or crop_around(idx, n_crops)):
                n_crops += 1

    stats["empty_bins"] = bins
    return stats


def tile_dataset(input_dir: str | Path, output_dir: str | Path, splits=("train", "val"),
                 settings: TileSettings | None = None, dataset_fraction: float = 1.0,
                 seed: int | None = None, overwrite: bool = False) -> Path:
    """Tile the splits of a YOLO dataset. Returns the folder of the tiled dataset.

    dataset_fraction < 1 tiles only the first fraction of each split's images
    (for a quick test). seed makes the random tile positions repeatable.
    An output folder that already holds files is refused, or with overwrite
    deleted first.
    """
    s = settings or TileSettings()
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not 0 < dataset_fraction <= 1.0:
        raise SystemExit("dataset_fraction must be in the range (0, 1].")
    if not input_dir.is_dir():
        raise SystemExit(f"Dataset folder not found: {input_dir}")
    if output_dir.resolve() == input_dir.resolve():
        raise SystemExit("The output folder is the input dataset: the tiles must go to a folder of their own.")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise SystemExit(f"Output folder already exists and is not empty: {output_dir}\n"
                             f"Use overwrite to delete it and tile again.")
        print(f"Clearing the existing output folder: {output_dir}")
        shutil.rmtree(output_dir)

    print(f"Tiling {input_dir}\n  splits={list(splits)}, dataset_fraction={dataset_fraction}, crop_size={s.crop_size}, "
          f"empty_fraction={s.empty_fraction}, empty_tries={s.empty_tries}, seed={seed}")
    rng = random.Random(seed)
    totals = {"images": 0, "instances_in": 0, "tiles": 0, "instances": 0, "empty_bins": [0] * 9}
    for split in splits:
        image_dir, label_dir = split_dirs(input_dir, split)
        image_paths = list_images(image_dir)
        if not image_paths:
            raise SystemExit(f"No .jpg / .png images found for the '{split}' split in {image_dir}")
        n_all = len(image_paths)
        if dataset_fraction < 1.0:
            image_paths = image_paths[:max(1, int(round(n_all * dataset_fraction)))]
        instances_in = sum(len(read_yolo_boxes(label_dir / f"{p.stem}.txt")) for p in image_paths)
        print(f"\n[{split}] {image_dir}\n[{split}] tiling {len(image_paths)} of {n_all} images, "
              f"{instances_in} object instances")

        stats = tile_split(image_paths, label_dir, output_dir / "images" / split, output_dir / "labels" / split,
                           s, rng, desc=f"{split} images")
        print(f"[{split}] saved {stats['tiles']} tiles with {stats['instances']} object instances "
              f"({stats['empty_saved']} empty tiles, {stats['empty_skipped']} skipped: no object-free tile found)")
        print(f"[{split}] empty tiles per 3x3 grid cell: {stats['empty_bins']}")
        totals["images"] += len(image_paths)
        totals["instances_in"] += instances_in
        totals["tiles"] += stats["tiles"]
        totals["instances"] += stats["instances"]
        totals["empty_bins"] = [a + b for a, b in zip(totals["empty_bins"], stats["empty_bins"])]

    print(f"\nTotal: {totals['images']} images / {totals['instances_in']} object instances -> "
          f"{totals['tiles']} tiles / {totals['instances']} object instances")
    print(f"Empty tiles per 3x3 grid cell, all splits: {totals['empty_bins']}")

    # Carry the class names over, so the tiles can be trained on directly.
    source_yaml = input_dir / "data.yaml"
    if source_yaml.is_file():
        source = yaml.safe_load(source_yaml.read_text()) or {}
        data = {**{split: f"images/{split}" for split in splits},
                **{k: source[k] for k in ("nc", "names") if k in source}}
        (output_dir / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    else:
        print(f"No data.yaml in {input_dir}: write one for the tiles before training on them.")
    return output_dir
