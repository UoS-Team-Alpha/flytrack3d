"""Draw YOLO labels on images, and view random samples of a YOLO dataset to check its labels.

draw_yolo_boxes paints the boxes into the image (OpenCV); plot_yolo_boxes outlines them on
matplotlib axes instead, where the lines stay thin however far the axes are zoomed in.
"""

from __future__ import annotations

import random
from pathlib import Path

from common.yolo_format import list_images, read_yolo_boxes, split_dirs

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


def plot_yolo_boxes(ax, image_path: str | Path, boxes: list, number: int | None = None,
                    colour: tuple = tuple(c / 255 for c in GREEN[::-1])) -> None:
    """Show an image on matplotlib axes with its normalised YOLO boxes outlined. A box is
    ([cls,] x_c, y_c, w, h): only its last four values are used. number = a label for the image,
    written in its top-left corner (to refer to it in a grid)."""
    from matplotlib.patches import Rectangle
    from PIL import Image

    img = Image.open(image_path)
    img_w, img_h = img.size
    ax.imshow(img)
    for *_, x_c, y_c, w, h in boxes:
        ax.add_patch(Rectangle(((x_c - w / 2) * img_w, (y_c - h / 2) * img_h), w * img_w, h * img_h,
                               fill=False, edgecolor=colour, linewidth=1.5))
    if number is not None:
        ax.text(0.02, 0.98, str(number), transform=ax.transAxes, va="top", ha="left", fontsize=12,
                fontweight="bold", color="white", bbox=dict(facecolor="black", alpha=0.7, pad=2))
    ax.set_xticks([])
    ax.set_yticks([])


def zoom_to_box(ax, image_path: str | Path, box, context: float = 4, min_px: float = 64) -> None:
    """Zoom axes showing an image onto one of its boxes ([cls,] x_c, y_c, w, h), leaving `context`
    times the box's size around it (at least min_px)."""
    from PIL import Image

    with Image.open(image_path) as im:   # header-only read, fast
        img_w, img_h = im.size
    *_, x_c, y_c, w, h = box
    half = max(context * max(w * img_w, h * img_h), min_px) / 2
    ax.set_xlim(x_c * img_w - half, x_c * img_w + half)
    ax.set_ylim(y_c * img_h + half, y_c * img_h - half)


def show_dataset_samples(dataset_dir: str | Path, split: str = "train", num_samples: int = 5,
                         class_names: dict | list | None = None, seed: int | None = None) -> None:
    """Open random images of a split one at a time, with their boxes drawn (any key or closing
    the window = next image, q or Esc = stop)."""
    import cv2

    image_dir, label_dir = split_dirs(dataset_dir, split)
    image_paths = list_images(image_dir)
    if not image_paths:
        print(f"No images found in {image_dir} for visualisation.")
        return

    samples = random.Random(seed).sample(image_paths, min(num_samples, len(image_paths)))
    print(f"\nShowing {len(samples)} sample images of the '{split}' split (press any key on the window or close it for the next, q or Esc to stop)...")
    for image_path in samples:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Cannot read {image_path.name}, skipped")
            continue
        boxes = read_yolo_boxes(label_dir / f"{image_path.stem}.txt")
        title = f"{image_path.stem} | Boxes: {len(boxes)}"
        print(f"Showing: {title}")
        cv2.imshow(title, draw_yolo_boxes(image, boxes, class_names))
        key = _wait_key_or_close(title)
        if key in (ord("q"), 27):   # q or Esc
            break
    cv2.destroyAllWindows()


def _wait_key_or_close(title: str) -> int:
    """Wait until a key is pressed on an OpenCV window or the window is closed, then close it.
    Returns the key, or -1 if the window was closed (cv2.waitKey(0) alone would then wait forever)."""
    import cv2

    key = -1
    try:
        while key == -1 and cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1:
            key = cv2.waitKey(50)
        cv2.destroyWindow(title)
    except cv2.error:   # the window is already gone
        pass
    return key & 0xFF if key != -1 else -1
