"""Find the green leaves (the tape star around the fruit) in one camera view by HSV thresholding.

    1. The view's median frame (arena_plotting/frames.py).
    2. Green mask in HSV (OpenCV hue 0-179), opened to drop specks and closed to join
       the tape strips across their creases and across the fruit.
    3. The leaves are the outer contour that contains the fruit's reprojected centre
       (else the nearest one within max_fruit_gap_px): the star is centred on the fruit.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DEFAULT_HSV = {
    "hue_range": [40, 90],   # OpenCV hue, 0-179
    "min_saturation": 80,
    "min_value": 40,
    "open_px": 5,
    "close_px": 15,
    "max_fruit_gap_px": 40,
}


def green_mask(bgr: np.ndarray, s: dict) -> np.ndarray:
    """uint8 0/1 mask of saturated green pixels, cleaned by an open and a close."""
    h, sat, val = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
    lo, hi = s["hue_range"]
    mask = ((h >= lo) & (h <= hi) & (sat >= s["min_saturation"]) & (val >= s["min_value"])).astype(np.uint8)
    if s["open_px"]:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((s["open_px"],) * 2, np.uint8))
    if s["close_px"]:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((s["close_px"],) * 2, np.uint8))
    return mask


def leaves_contour(mask: np.ndarray, fruit_xy: np.ndarray, s: dict) -> np.ndarray | None:
    """(N, 2) raw-pixel outline of the green region around the fruit, or None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    pt = (float(fruit_xy[0]), float(fruit_xy[1]))
    signed = [cv2.pointPolygonTest(c, pt, True) for c in contours]   # > 0 inside, < 0 outside
    inside = [c for c, d in zip(contours, signed) if d >= 0]
    if inside:
        best = max(inside, key=cv2.contourArea)
    else:
        i = int(np.argmax(signed))
        if -signed[i] > s["max_fruit_gap_px"]:
            return None
        best = contours[i]
    return best[:, 0].astype(np.float64)


def save_debug_image(bgr: np.ndarray, mask: np.ndarray, contour: np.ndarray | None, fruit_xy: np.ndarray,
                     path: Path, label: str) -> None:
    """Median frame with the green mask tinted magenta, the leaves outline used in yellow."""
    img = bgr.copy()
    img[mask > 0] = (0.5 * img[mask > 0] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
    if contour is not None:
        cv2.polylines(img, [np.round(contour).astype(np.int32)], True, (0, 255, 255), 2)
    cv2.drawMarker(img, tuple(int(v) for v in np.round(fruit_xy)), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.putText(img, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
