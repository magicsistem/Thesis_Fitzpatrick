#!/usr/bin/env python3
"""Manual clean-skin benchmark; excluded from CI."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import thesis_fitzpatrick.masks as masks  # noqa: E402


def reference(image: Image.Image, lesion: np.ndarray) -> None:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    margin = max(3, round(min(width, height) * 0.015))
    outer = max(margin + 3, round(min(width, height) * 0.22))
    exclusion = masks._dilate(lesion, margin)
    ring = masks._dilate(lesion, outer) & ~exclusion
    maximum, minimum = rgb.max(axis=2), rgb.min(axis=2)
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    valid = maximum > 8
    highlight = masks._dilate((maximum >= 248) & ((maximum - minimum) <= 22), max(1, margin // 2))
    local_max = np.asarray(Image.fromarray(gray).filter(ImageFilter.MaxFilter(masks._odd_size(max(2, margin)))))
    hair = masks._dilate(((local_max.astype(np.int16) - gray.astype(np.int16)) >= 45) & (gray <= 135), max(1, margin // 3))
    clean = ring & valid & ~highlight & ~hair
    if int(clean.sum()) < max(256, round(width * height * 0.005)):
        clean = valid & ~exclusion & ~highlight & ~hair


def timed(function, image, lesions) -> float:
    start = time.perf_counter()
    for lesion in lesions:
        function(image, lesion)
    return time.perf_counter() - start


if __name__ == "__main__":
    rng = np.random.default_rng(17)
    rgb = np.full((192, 288, 3), (150, 105, 85), dtype=np.uint8)
    rgb[rng.random(rgb.shape[:2]) < 0.01] = (35, 35, 35)
    image = Image.fromarray(rgb)
    lesions = []
    for index in range(16):
        lesion = np.zeros(rgb.shape[:2], dtype=bool)
        y, x = 32 + index * 23, 80 + index * 31
        lesion[y:y + 80, x:x + 90] = True
        lesions.append(lesion)
    context = masks.prepare_clean_skin_context(image)
    timings = {"pil_reference": [], "opencv_no_context": [], "opencv_context": []}
    for _ in range(3):
        timings["pil_reference"].append(timed(reference, image, lesions))
        timings["opencv_no_context"].append(timed(masks.build_clean_skin_mask, image, lesions))
        timings["opencv_context"].append(timed(lambda img, lesion: masks.build_clean_skin_mask(img, lesion, context=context), image, lesions))
    for name, values in timings.items():
        print(f"{name}_median_seconds={statistics.median(values):.6f}")
    print(f"speedup_no_context={statistics.median(timings['pil_reference']) / statistics.median(timings['opencv_no_context']):.2f}x")
    print(f"speedup_context={statistics.median(timings['pil_reference']) / statistics.median(timings['opencv_context']):.2f}x")
