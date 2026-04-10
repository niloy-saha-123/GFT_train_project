from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np

from .types import CameraProfile


def apply_ignore_rectangles(frame: np.ndarray, rectangles: Iterable[Tuple[int, int, int, int]]) -> np.ndarray:
    out = frame.copy()
    for x, y, w, h in rectangles:
        cv2.rectangle(out, (x, y), (x + w, y + h), color=(0, 0, 0), thickness=-1)
    return out


def apply_roi_mask(frame: np.ndarray, polygon: list[tuple[float, float]]) -> np.ndarray:
    if not polygon:
        return frame
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    pts = np.array([[int(x), int(y)] for x, y in polygon], dtype=np.int32)
    cv2.fillPoly(mask, [pts], color=255)
    return cv2.bitwise_and(frame, frame, mask=mask)


def preprocess_frame(frame: np.ndarray, profile: CameraProfile) -> np.ndarray:
    out = apply_roi_mask(frame, profile.roi_polygon)
    out = apply_ignore_rectangles(out, profile.ignore_rectangles)
    return out
