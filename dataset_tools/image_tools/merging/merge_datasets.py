"""Merge a folder of YOLO datasets into one, without ever putting the same image in it twice.

Every sub-folder of the input folder that holds a split (images/<split> + labels/<split>,
or <split>/images + <split>/labels) is one dataset to merge. Each of its label files with
an image is copied to the output as

    <output_dir>/images/<split>/<dataset>_<image>.<ext>
    <output_dir>/labels/<split>/<dataset>_<image>.txt
    <output_dir>/<split>.txt     the images of the split, one per line
    <output_dir>/data.yaml

The output may already hold a dataset, which is then added to. An image is only added when
no image with the same content is in the output yet: the images already there (in any
split) are hashed first, and every image to add is hashed and compared with them and with
the ones added before it. So the same frame in two of the datasets, a dataset merged a
second time, or a frame in train of one dataset and val of another, all end up in the
output once. The hash is of the whole file, so two images are only the same when their
files are byte for byte identical; the same picture saved again (e.g. as .jpg and as .png)
is not recognised. A file already in the output is never overwritten.

keep_classes keeps only the boxes of those classes, renumbered from 0 in the order given.
The datasets to merge are never changed.
"""

from __future__ import annotations

import hashlib
import shutil
from collections import Counter
from pathlib import Path

import yaml
from tqdm import tqdm

from common.datasets import read_class_names
from common.yolo_format import SPLIT_NAMES, find_splits, split_dirs

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}   # compared in lower case
MAX_EXAMPLES = 5                             # how many files a warning lists


def file_hash(path: Path) -> str:
    """Hash of the whole file: equal for two files only when their content is identical."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "blake2b").hexdigest()


def images_in(image_dir: Path) -> list[Path]:
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def output_dirs(output_dir: Path, split: str) -> tuple[Path, Path]:
    """(image folder, label folder) of a split of the output: the layout it already has, else images/<split>."""
    if (output_dir / split / "images").is_dir() and not (output_dir / "images" / split).is_dir():
        return output_dir / split / "images", output_dir / split / "labels"
    return output_dir / "images" / split, output_dir / "labels" / split


def label_rows(text: str, class_map: dict[int, int] | None) -> list[str]:
    """The rows of a label file, kept and renumbered by class_map ({class id: new id}; None = every row as it is)."""
    rows = [line.split() for line in text.splitlines() if line.strip()]
    if class_map is None:
        return [" ".join(row) for row in rows]
    return [" ".join([str(class_map[int(row[0])]), *row[1:]]) for row in rows if int(row[0]) in class_map]


def resolve_classes(sources: list[Path], keep_classes: list[int] | None,
                    class_names: list[str] | None) -> tuple[dict[int, int] | None, dict[int, str]]:
    """(class_map, {class id in the output: name}). The names come from class_names, else from the
    data.yaml files of the datasets (the first one that names a class wins)."""
    class_map = {int(old): new for new, old in enumerate(keep_classes)} if keep_classes else None
    if class_names:
        if class_map and len(class_names) != len(class_map):
            raise SystemExit(f"class_names has {len(class_names)} names for the {len(class_map)} classes of keep_classes.")
        return class_map, dict(enumerate(class_names))

    source_names = {d.name: read_class_names(d / "data.yaml") for d in sources if (d / "data.yaml").is_file()}
    ids = list(class_map) if class_map else sorted({i for names in source_names.values() for i in names})
    names = {}
    for class_id in ids:
        found = {dataset: n[class_id] for dataset, n in source_names.items() if class_id in n}
        if not found:
            break
        if len(set(found.values())) > 1:
            print(f"WARNING: class {class_id} has different names in the datasets: "
                  f"{', '.join(f'{name!r} in {dataset}' for dataset, name in found.items())}. Using the first.")
        names[class_map[class_id] if class_map else class_id] = next(iter(found.values()))
    if not ids or len(names) < len(ids):
        raise SystemExit("The datasets do not name every class to keep (no data.yaml, or the class is not in it): "
                         "give class_names.")
    return class_map, names


def hash_output(output_dir: Path) -> tuple[dict[str, tuple[Path, Path]], list[tuple[Path, Path]]]:
    """({hash: (image, its label file)} of the images already in the output, [(image, the earlier image it repeats)])."""
    existing = [(image, labels / f"{image.stem}.txt") for split in find_splits(output_dir)
                for images, labels in [output_dirs(output_dir, split)] for image in images_in(images)]
    seen, repeats = {}, []
    for image, label in tqdm(existing, desc="Hashing the output's images", unit="img", disable=not existing):
        digest = file_hash(image)
        if digest in seen:
            repeats.append((image, seen[digest][0]))
        else:
            seen[digest] = (image, label)
    return seen, repeats


def print_examples(pairs: list[tuple[Path, Path]], image_base: Path, output_dir: Path) -> None:
    """List (image, the image of the output it is the same as) pairs, as paths in image_base and in the output."""
    for image, kept in pairs[:MAX_EXAMPLES]:
        print(f"    {image.relative_to(image_base)}  =  {kept.relative_to(output_dir)}")
    if len(pairs) > MAX_EXAMPLES:
        print(f"    ... and {len(pairs) - MAX_EXAMPLES} more")


def write_lists_and_yaml(output_dir: Path, names: dict[int, str]) -> Path | None:
    """Write <split>.txt for every split of the output and its data.yaml. Returns the data.yaml; None if no images."""
    lists = {}
    for split in find_splits(output_dir):
        images = images_in(output_dirs(output_dir, split)[0])
        if images:
            # "./" makes Ultralytics read the lines relative to the list file, not to the working directory.
            (output_dir / f"{split}.txt").write_text(
                "".join(f"./{p.relative_to(output_dir).as_posix()}\n" for p in images), encoding="utf-8")
            lists["val" if split in ("valid", "validation") else split] = f"{split}.txt"
    if not lists:
        return None
    first = next(iter(lists.values()))   # a dataset with one split trains and validates on it
    data = {"train": lists.get("train", first), "val": lists.get("val", first)}
    if "test" in lists:
        data["test"] = lists["test"]
    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({**data, "nc": len(names), "names": names}, sort_keys=False), encoding="utf-8")
    return data_yaml


def merge_datasets(input_dir: str | Path, output_dir: str | Path, splits: list[str] | None = None,
                   keep_classes: list[int] | None = None, class_names: list[str] | None = None,
                   keep_empty: bool = True) -> Path:
    """Merge the YOLO datasets in input_dir into output_dir. Returns the path of the output's data.yaml.

    splits = the splits to merge; None = every one found. keep_classes = the class ids to keep,
    renumbered from 0 in that order; None = every class as it is. class_names = the names of the
    output's classes; None = read them from the datasets. keep_empty = also add images whose
    label file is empty (or has no box of a kept class), as background images.
    """
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input folder not found: {input_dir}")
    # The output is left out, so a merged dataset that sits in the input folder is not merged into itself.
    sources = [d for d in sorted(input_dir.iterdir()) if d.is_dir() and d != output_dir and find_splits(d)]
    if not sources:
        raise SystemExit(f"No dataset found in {input_dir}: its sub-folders need images/<split>/ and labels/<split>/.")
    splits = list(splits) if splits else [s for s in SPLIT_NAMES if any(s in find_splits(d) for d in sources)]

    class_map, names = resolve_classes(sources, keep_classes, class_names)
    if (output_dir / "data.yaml").is_file() and (existing := read_class_names(output_dir / "data.yaml")) != names:
        raise SystemExit(f"The output already holds a dataset with other classes ({existing}, to merge: {names}): "
                         f"{output_dir}")

    seen, repeats = hash_output(output_dir)
    print(f"Merging {len(sources)} datasets ({', '.join(splits)}) into {output_dir}, which holds "
          f"{len(seen) + len(repeats)} images.")
    if repeats:
        print(f"WARNING: {len(repeats)} images already in the output repeat another one of it (left as they are):")
        print_examples(repeats, output_dir, output_dir)

    stats = {d.name: Counter() for d in sources}
    duplicates, label_conflicts, name_clashes = [], [], []
    for split in splits:
        out_images, out_labels = output_dirs(output_dir, split)
        used = {p.stem for p in out_images.iterdir()} if out_images.is_dir() else set()
        for source in sources:
            image_dir, label_dir = split_dirs(source, split)
            if not image_dir.is_dir() or not label_dir.is_dir():
                continue
            out_images.mkdir(parents=True, exist_ok=True)
            out_labels.mkdir(parents=True, exist_ok=True)
            images = {p.stem: p for p in images_in(image_dir)}
            count = stats[source.name]
            for label_path in tqdm(sorted(label_dir.glob("*.txt")), desc=f"{source.name} {split}", unit="img"):
                image_path = images.get(label_path.stem)
                if image_path is None:
                    count["no_image"] += 1
                    continue
                rows = label_rows(label_path.read_text(), class_map)
                if not rows and not keep_empty:
                    count["empty"] += 1
                    continue

                digest = file_hash(image_path)
                if digest in seen:   # the same image is in the output already: never add it again
                    kept_image, kept_label = seen[digest]
                    count["duplicate"] += 1
                    duplicates.append((image_path, kept_image))
                    kept_rows = label_rows(kept_label.read_text(), None) if kept_label.is_file() else []
                    if sorted(rows) != sorted(kept_rows):
                        label_conflicts.append((image_path, kept_image))
                    continue

                name = f"{source.name}_{label_path.stem}"
                if name in used:     # another image has this name in the output: never overwrite it
                    count["name_clash"] += 1
                    name_clashes.append(image_path)
                    continue
                target = out_images / f"{name}{image_path.suffix}"
                (out_labels / f"{name}.txt").write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")
                shutil.copy2(image_path, target)
                seen[digest] = (target, out_labels / f"{name}.txt")
                used.add(name)
                count["added"] += 1

    data_yaml = write_lists_and_yaml(output_dir, names)

    columns = ["added", "duplicate", "empty", "no_image", "name_clash"]
    width = max(len(name) for name in [*stats, "dataset"])
    print(f"\n{'dataset':<{width}}" + "".join(f"{c:>12}" for c in columns))
    for name, count in stats.items():
        print(f"{name:<{width}}" + "".join(f"{count[c]:>12}" for c in columns))
    print(f"{'total':<{width}}" + "".join(f"{sum(count[c] for count in stats.values()):>12}" for c in columns))
    print("  duplicate = same image already in the output, empty = no box kept (keep_empty is off), "
          "no_image = label file without an image")

    if duplicates:
        print(f"\n{len(duplicates)} duplicate images were not added:")
        print_examples(duplicates, input_dir, output_dir)
    if label_conflicts:
        print(f"\nWARNING: {len(label_conflicts)} of the duplicates have other labels than the copy in the output, "
              f"whose labels were kept:")
        print_examples(label_conflicts, input_dir, output_dir)
    if name_clashes:
        print(f"\nWARNING: {len(name_clashes)} images were not added, as a different image has their name in the output:")
        for image in name_clashes[:MAX_EXAMPLES]:
            print(f"    {image.relative_to(input_dir)}")

    if data_yaml is None:
        raise SystemExit("\nNothing was merged: the output holds no images.")
    print(f"\nThe output now holds {len(seen) + len(repeats)} images, {len(seen)} of them different.")
    return data_yaml
