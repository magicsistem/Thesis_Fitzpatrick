#!/usr/bin/env python3
"""Manual benchmark for the connected-component point scan; excluded from CI."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from thesis_fitzpatrick.preprocessing import _component_points  # noqa: E402


def reference(labels: np.ndarray, _stats: np.ndarray, label: int) -> np.ndarray | None:
    points = np.column_stack(np.where(labels == label))[:, ::-1].astype(np.float32)
    return points if len(points) >= 3 else None


def measure(function, labels: np.ndarray, stats: np.ndarray, count: int) -> float:
    started = time.perf_counter()
    for label in range(1, count):
        function(labels, stats, label)
    return time.perf_counter() - started


if __name__ == "__main__":
    candidate = np.zeros((1024, 1024), np.uint8)
    for y in range(8, 1016, 16):
        for x in range(8, 1016, 16):
            candidate[y:y + 3, x:x + 3] = 1
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    old = measure(reference, labels, stats, count)
    new = measure(_component_points, labels, stats, count)
    print(f"components={count - 1} old_seconds={old:.6f} optimized_seconds={new:.6f} speedup={old / new:.2f}x")
