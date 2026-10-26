#!/usr/bin/env python3
"""Object detectors for the evaluation protocol (where did the user put the object?).

A detector takes a BGR image (OpenCV) and returns the object's pixel position (x, y) or None.
Presets choose one by name ("detector" in omx_presets/<id>.json):
  red_hsv  red blobs in HSV (strawberries)
  none     no detection (objects without a simple color, e.g. duty-free boxes and bags)
To add one: write `def my_detector(img) -> tuple[int, int] | None` and register it in DETECTORS.
A preset can limit detection to a region ("detect_roi": [x0, y0, x1, y1]), e.g. the mat without the
robot base, whose red motor parts are always at the top of the front view.
"""

from __future__ import annotations

import cv2
import numpy as np

MIN_AREA = 150  # px^2: smaller red blobs (glints, cable ties) count as "nothing detected"


def red_hsv(img: np.ndarray) -> tuple[int, int] | None:
    """Centre of the largest saturated red blob (strawberry, possibly merged with the red tape under it)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 120, 60), (8, 255, 255)) | cv2.inRange(hsv, (170, 120, 60), (180, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (int(centroids[i][0]), int(centroids[i][1])) if stats[i, cv2.CC_STAT_AREA] > MIN_AREA else None


DETECTORS = {
    "red_hsv": red_hsv,
    "none": lambda img: None,
}


def detect(name: str, img: np.ndarray | None, roi: list | None = None) -> tuple[int, int] | None:
    """Detect inside roi (x0, y0, x1, y1; whole image if None); the result is in full-image pixels."""
    if img is None or name not in DETECTORS:
        return None
    x0 = y0 = 0
    if roi:
        h, w = img.shape[:2]
        x0, y0 = max(0, int(roi[0])), max(0, int(roi[1]))
        x1, y1 = min(w, int(roi[2])), min(h, int(roi[3]))
        if x1 <= x0 or y1 <= y0:
            return None
        img = img[y0:y1, x0:x1]
    p = DETECTORS[name](img)
    return (p[0] + x0, p[1] + y0) if p else None


def detect_jpeg(name: str, data: bytes, roi: list | None = None) -> tuple[int, int] | None:
    if name == "none" or not data:
        return None
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return detect(name, img, roi)
