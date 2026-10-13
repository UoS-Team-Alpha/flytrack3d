"""Convert a COCO detection dataset to YOLO format with Ultralytics.

The COCO dataset is the layout convert_yolo_to_coco writes:

    <coco_dir>/data/                               the images of every split
    <coco_dir>/annotations/instances_<split>.json

Ultralytics' convert_coco writes the labels only, so the images and a data.yaml
are added here, giving a dataset Ultralytics can train from:

    <export_dir>/images/<split>/
    <export_dir>/labels/<split>/
    <export_dir>/data.yaml

YOLO class = COCO category id - 1 (that is what convert_coco does), so the
category ids must be 1..N. Images without annotations get no label file, which
YOLO reads as background images.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import yaml


def read_categories(json_files: list[Path]) -> dict[int, str]:
    """{YOLO class: name} shared by the annotation files; their category ids must be 1..N."""
    names = None
    for json_file in json_files:
        with open(json_file, encoding="utf-8") as f:
            categories = {c["id"]: c["name"] for c in json.load(f)["categories"]}
        if sorted(categories) != list(range(1, len(categories) + 1)):
            raise SystemExit(f"{json_file.name}: category ids must be 1..N to convert to YOLO classes, "
                             f"got {sorted(categories)}")
        if names is not None and categories != names:
            raise SystemExit(f"{json_file.name}: categories differ from the other annotation files.")
        names = categories
    return {i - 1: name for i, name in sorted(names.items())}


def add_images(json_file: Path, images_dir: Path, split_dir: Path, copy_images: bool) -> int:
    """Copy (or symlink) the images listed in a COCO json into split_dir. Returns how many."""
    with open(json_file, encoding="utf-8") as f:
        file_names = [img["file_name"] for img in json.load(f)["images"]]
    for file_name in file_names:
        source, target = images_dir / file_name, split_dir / file_name
        if not source.is_file():
            raise SystemExit(f"Image listed in {json_file.name} not found: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if copy_images:
            shutil.copy2(source, target)
        else:
            os.symlink(source.resolve(), target)
    return len(file_names)


def convert_coco_to_yolo(coco_dir: str | Path, export_dir: str | Path, splits=None,
                         images_dir: str | Path | None = None, copy_images: bool = True) -> Path:
    """Convert a COCO dataset to YOLO. Returns the path of the new data.yaml.

    splits = the instances_<split>.json files to convert; None = every one in
    <coco_dir>/annotations. images_dir defaults to <coco_dir>/data.
    copy_images=False symlinks the images instead of copying them.
    """
    from ultralytics.data.converter import convert_coco   # imported here: only this conversion needs it

    coco_dir, export_dir = Path(coco_dir), Path(export_dir)
    annotations_dir = coco_dir / "annotations"
    images_dir = Path(images_dir) if images_dir else coco_dir / "data"
    if splits is None:
        json_files = sorted(annotations_dir.glob("instances_*.json"))
    else:
        json_files = [annotations_dir / f"instances_{split}.json" for split in splits]
    missing = [str(f) for f in json_files if not f.is_file()]
    if missing or not json_files:
        raise SystemExit(f"COCO annotation files not found: {missing or annotations_dir / 'instances_*.json'}")
    # convert_coco writes to a renamed folder (<export_dir>2, ...) if the folder exists, so refuse instead.
    if export_dir.exists():
        if any(export_dir.iterdir()):
            raise SystemExit(f"Output folder already exists and is not empty: {export_dir}")
        export_dir.rmdir()

    names = read_categories(json_files)

    # convert_coco converts every json in a folder, so give it a folder holding only the chosen splits.
    selected_dir = export_dir.with_name(export_dir.name + "_annotations_tmp")
    selected_dir.mkdir(parents=True)
    try:
        for json_file in json_files:
            os.symlink(json_file.resolve(), selected_dir / json_file.name)
        # cls91to80=False: keep our own classes rather than remapping the 91 COCO ids to the 80 COCO classes.
        convert_coco(labels_dir=str(selected_dir), save_dir=str(export_dir), cls91to80=False)
    finally:
        shutil.rmtree(selected_dir)

    split_names = [f.stem.replace("instances_", "") for f in json_files]
    for split, json_file in zip(split_names, json_files):
        n = add_images(json_file, images_dir, export_dir / "images" / split, copy_images)
        print(f"{'Copied' if copy_images else 'Linked'} {n} '{split}' images.")

    data_yaml = export_dir / "data.yaml"
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump({"path": str(export_dir.resolve()),
                        **{split: f"images/{split}" for split in split_names},
                        "nc": len(names), "names": names}, f, sort_keys=False)
    return data_yaml
