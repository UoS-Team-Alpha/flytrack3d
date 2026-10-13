"""Rebuild a YOLO dataset from a CVAT annotation export and the video it was annotated on.

A CVAT export of a video task holds the label files but not the images. This
reads the labels straight from the export zip, pulls the matching frames out of
the video and writes a dataset Ultralytics can train from:

    <output_dir>/images/train/frame_<n>.png
    <output_dir>/labels/train/frame_<n>.txt
    <output_dir>/train.txt       the images, one per line
    <output_dir>/data.yaml       train and val both point at train.txt

A label's frame number is the number at the end of its file name
(frame_000123.txt = frame 123) and is taken as the frame index in the video, so
the task must have been made from this video with frame step 1. Only frames
with annotations are kept, unless keep_empty.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path, PurePosixPath

import yaml
from tqdm import tqdm

SPLIT_LISTS = {"train.txt", "val.txt", "test.txt"}   # image lists in the export, not labels


def read_labels(export: zipfile.ZipFile, keep_empty: bool = False) -> dict[int, str]:
    """{frame number: label text} of the export. Empty label files are dropped unless keep_empty."""
    members = [PurePosixPath(n) for n in export.namelist()
               if n.endswith(".txt") and PurePosixPath(n).name not in SPLIT_LISTS]
    if not members:
        raise SystemExit("No label .txt files found in the annotation zip.")
    # The labels are in labels/train of a YOLO export; otherwise use the folder of the first label found.
    folders = {m.parent for m in members}
    folder = next((f for f in sorted(folders) if f.as_posix().endswith("labels/train")), members[0].parent)

    labels = {}
    for member in members:
        number = re.search(r"(\d+)$", member.stem)
        if member.parent != folder or not number:
            continue
        text = export.read(member.as_posix()).decode("utf-8")
        if text.strip() or keep_empty:
            labels[int(number.group(1))] = text
    return labels


def read_class_names(export: zipfile.ZipFile) -> list[str] | None:
    """Class names from the export's data.yaml or obj.names; None if it has neither."""
    for name in export.namelist():
        if PurePosixPath(name).name == "data.yaml":
            names = (yaml.safe_load(export.read(name)) or {}).get("names")
            if isinstance(names, dict):
                return [names[k] for k in sorted(names)]
            if names:
                return list(names)
        if PurePosixPath(name).name == "obj.names":
            return [line.strip() for line in export.read(name).decode("utf-8").splitlines() if line.strip()]
    return None


def reconstruct_cvat_dataset(annotations_zip: str | Path, video: str | Path, output_dir: str | Path,
                             class_names: list[str] | None = None, keep_empty: bool = False) -> Path:
    """Write the labelled frames of a video as a YOLO dataset. Returns the path of its data.yaml.

    class_names = the classes, in label order; None = read them from the export
    (data.yaml or obj.names). keep_empty also keeps frames whose label file is
    empty, as background images.
    """
    import cv2   # imported here: only this tool needs OpenCV

    annotations_zip, video, output_dir = Path(annotations_zip), Path(video), Path(output_dir)
    if not annotations_zip.is_file():
        raise SystemExit(f"Annotation zip not found: {annotations_zip}")
    if not video.is_file():
        raise SystemExit(f"Video not found: {video}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Output folder already exists and is not empty: {output_dir}")

    with zipfile.ZipFile(annotations_zip) as export:
        labels = read_labels(export, keep_empty)
        class_names = class_names or read_class_names(export)
    if not labels:
        raise SystemExit("No annotations found in the zip (every label file is empty).")
    if not class_names:
        raise SystemExit("The export names no classes (no data.yaml or obj.names): give class_names.")
    first, last = min(labels), max(labels)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open the video: {video}")
    n_frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    print(f"Video:  {n_frames} frames at {fps:.2f} FPS")
    print(f"Labels: {len(labels)} frames, numbered {first} to {last}")
    if last >= n_frames:
        print(f"WARNING: the highest labelled frame ({last}) is beyond the end of the video ({n_frames} frames).\n"
              f"         Check that this is the annotated video and that the CVAT frame step was 1.")

    images_dir, labels_dir = output_dir / "images" / "train", output_dir / "labels" / "train"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    # Seek to the first labelled frame, then read every frame up to the last one in order.
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    saved = []
    for index in tqdm(range(first, last + 1), desc="Extracting frames"):
        ok, frame = cap.read()
        if not ok:
            break
        if index in labels:
            name = f"frame_{index:06d}"
            cv2.imwrite(str(images_dir / f"{name}.png"), frame)
            (labels_dir / f"{name}.txt").write_text(labels[index], encoding="utf-8")
            saved.append(f"images/train/{name}.png")
    cap.release()

    (output_dir / "train.txt").write_text("\n".join(saved) + "\n", encoding="utf-8")
    data_yaml = output_dir / "data.yaml"
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump({"train": "train.txt", "val": "train.txt", "nc": len(class_names),
                        "names": dict(enumerate(class_names))}, f, sort_keys=False)

    print(f"\nSaved {len(saved)} of {len(labels)} labelled frames.")
    if len(saved) < len(labels):
        print(f"WARNING: {len(labels) - len(saved)} labelled frames could not be read from the video.")
    return data_yaml
