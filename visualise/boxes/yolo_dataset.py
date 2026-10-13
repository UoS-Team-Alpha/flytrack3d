"""Draw YOLO labels on images, and view random samples of a YOLO dataset to check its labels."""

from __future__ import annotations

import random
from pathlib import Path

from dataset_tools.image_tools.yolo_format import list_images, read_yolo_boxes, split_dirs

GREEN = (0, 255, 0)   # BGR


def draw_yolo_boxes(image, boxes: list, class_names: dict | list | None = None, colour: tuple = GREEN):
    """Draw normalised YOLO boxes [cls, x_c, y_c, w, h] on an image (in place), each with its class above it."""
    import cv2

    img_h, img_w = image.shape[:2]
    for cls, x_c, y_c, w, h in boxes:
        x_min, y_min = int((x_c - w / 2) * img_w), int((y_c - h / 2) * img_h)
        x_max, y_max = int((x_c + w / 2) * img_w), int((y_c + h / 2) * img_h)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), colour, 2)
        label = str(class_names[int(cls)]) if class_names else f"Class {int(cls)}"
        cv2.putText(image, label, (x_min, max(15, y_min - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    return image


def show_dataset_samples(dataset_dir: str | Path, split: str = "train", num_samples: int = 5,
                         class_names: dict | list | None = None, seed: int | None = None) -> None:
    """Open random images of a split one at a time, with their boxes drawn (any key = next image)."""
    import cv2

    image_dir, label_dir = split_dirs(dataset_dir, split)
    image_paths = list_images(image_dir)
    if not image_paths:
        print(f"No images found in {image_dir} for visualisation.")
        return

    samples = random.Random(seed).sample(image_paths, min(num_samples, len(image_paths)))
    print(f"\nShowing {len(samples)} sample images of the '{split}' split (press any key on the window for the next)...")
    for image_path in samples:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Cannot read {image_path.name}, skipped")
            continue
        boxes = read_yolo_boxes(label_dir / f"{image_path.stem}.txt")
        title = f"{image_path.stem} | Boxes: {len(boxes)}"
        print(f"Showing: {title}")
        cv2.imshow(title, draw_yolo_boxes(image, boxes, class_names))
        cv2.waitKey(0)
        cv2.destroyWindow(title)
    cv2.destroyAllWindows()
