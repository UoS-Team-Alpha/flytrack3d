"""Convert a YOLO detection dataset to COCO format with FiftyOne.

The YOLO dataset is the folder Ultralytics trains from: a data.yaml naming the
classes and the splits (image folders or .txt lists of images), with a label
.txt next to each image under labels/. The COCO dataset is written as

    <export_dir>/data/                             the images of every split
    <export_dir>/annotations/instances_<split>.json

Boxes become absolute [x, y, width, height] pixels, and category ids start at 1
(YOLO class 0 = COCO category 1), in the order of the yaml's names.
"""

from __future__ import annotations

from pathlib import Path


def convert_yolo_to_coco(dataset_dir: str | Path, export_dir: str | Path, splits=("train", "val"),
                         yaml_path: str | Path | None = None, copy_images: bool = True) -> Path:
    """Convert the splits of a YOLO dataset to COCO. Returns the COCO dataset folder.

    yaml_path defaults to <dataset_dir>/data.yaml. copy_images=False symlinks the
    images into <export_dir>/data instead of copying them.
    """
    import fiftyone as fo   # imported here: only this conversion needs FiftyOne

    dataset_dir, export_dir = Path(dataset_dir), Path(export_dir)
    yaml_path = Path(yaml_path) if yaml_path else dataset_dir / "data.yaml"
    if not yaml_path.is_file():
        raise SystemExit(f"YOLO dataset yaml not found: {yaml_path}")
    if export_dir.exists() and any(export_dir.iterdir()):
        raise SystemExit(f"Output folder already exists and is not empty: {export_dir}")

    for split in splits:
        print(f"Loading YOLO '{split}' split...")
        dataset = fo.Dataset.from_dir(
            dataset_type=fo.types.YOLOv5Dataset,
            dataset_dir=str(dataset_dir),
            yaml_path=str(yaml_path),
            split=split,
        )

        print(f"Exporting '{split}' split ({len(dataset)} images) to COCO...")
        dataset.export(
            export_dir=str(export_dir),
            dataset_type=fo.types.COCODetectionDataset,
            labels_path=f"annotations/instances_{split}.json",
            label_field="ground_truth",
            export_media=True if copy_images else "symlink",
        )
        dataset.delete()   # drop it from FiftyOne's database; the files on disk are untouched
    return export_dir
