from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis_fitzpatrick.benchmark import ImageInput  # noqa: E402
from thesis_fitzpatrick.grabcut import (  # noqa: E402
    VALID_GRABCUT_LABELS,
    build_robust_trimap,
    classify_mask_failure,
    common_postprocess,
    grabcut_classic,
    grabcut_robust,
)
from thesis_fitzpatrick.preprocessing import BBox, Detection, run_p0  # noqa: E402


CONFIG = json.loads((REPO_ROOT / "configs" / "benchmark" / "default.json").read_text())


class CenterDetector:
    @property
    def identity(self) -> dict:
        return {"type": "synthetic-center"}

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        height, width = rgb.shape[:2]
        return [Detection(BBox(width // 3, height // 3, 2 * width // 3, 2 * height // 3), 0.95)]


def synthetic_image() -> np.ndarray:
    rgb = np.full((120, 160, 3), (185, 145, 120), dtype=np.uint8)
    cv2.ellipse(rgb, (80, 60), (25, 18), 0, 0, 360, (45, 30, 25), -1)
    return rgb


class GrabCutTests(unittest.TestCase):
    def test_classic_runs_without_yolo_or_ground_truth(self) -> None:
        result = grabcut_classic(synthetic_image(), margin_fraction=0.05, iterations=3)
        self.assertEqual(result.native_mask.shape, (120, 160))
        self.assertEqual(set(np.unique(result.native_mask)).difference({0, 1}), set())
        self.assertEqual(result.model_identity["mode"], "classic")
        self.assertFalse(result.details["ground_truth_used"])

    def test_robust_trimap_has_only_opencv_labels_and_safe_background(self) -> None:
        p0 = run_p0(ImageInput("one", "synthetic", "validation", synthetic_image()), CONFIG, CenterDetector())
        trimap = build_robust_trimap(p0)
        self.assertTrue(set(np.unique(trimap)).issubset(VALID_GRABCUT_LABELS))
        self.assertTrue(np.all(trimap[0] == cv2.GC_BGD))
        self.assertTrue(np.any(trimap == cv2.GC_PR_FGD))
        self.assertTrue(np.any(trimap == cv2.GC_PR_BGD))

    def test_outside_fov_is_always_sure_background(self) -> None:
        rgb = np.zeros((140, 180, 3), dtype=np.uint8)
        cv2.ellipse(rgb, (90, 70), (75, 55), 0, 0, 360, (175, 130, 105), -1)
        cv2.circle(rgb, (90, 70), 20, (40, 25, 20), -1)
        p0 = run_p0(ImageInput("one", "synthetic", "validation", rgb), CONFIG, CenterDetector())
        trimap = build_robust_trimap(p0)
        self.assertTrue(np.all(trimap[p0.roi_fov_mask == 0] == cv2.GC_BGD))

    def test_robust_is_distinct_and_uses_p0_roi(self) -> None:
        p0 = run_p0(ImageInput("one", "synthetic", "validation", synthetic_image()), CONFIG, CenterDetector())
        result = grabcut_robust(p0, iterations=3)
        self.assertEqual(result.model_identity["mode"], "robust")
        self.assertEqual(result.details["initialization"], "GC_INIT_WITH_MASK")
        self.assertFalse(result.details["ground_truth_used"])
        self.assertEqual(result.native_mask.shape, p0.roi_input.shape[:2])

    def test_empty_and_nearly_complete_masks_are_typed_failures(self) -> None:
        self.assertEqual(classify_mask_failure(np.zeros((10, 10), np.uint8))[0], "empty_mask")
        self.assertEqual(classify_mask_failure(np.ones((10, 10), np.uint8))[0], "nearly_complete_mask")

    def test_common_postprocess_intersects_fov_and_selects_bbox_component(self) -> None:
        raw = np.zeros((100, 120), np.uint8)
        cv2.circle(raw, (25, 50), 12, 1, -1)
        cv2.circle(raw, (90, 50), 15, 1, -1)
        raw[2:5, 2:5] = 1
        fov = np.ones_like(raw)
        fov[:, :10] = 0
        cleaned, details = common_postprocess(
            raw, fov, CONFIG["postprocessing"], selected_bbox=BBox(15, 35, 40, 65)
        )
        self.assertEqual(int(cleaned[50, 25]), 1)
        self.assertEqual(int(cleaned[50, 90]), 0)
        self.assertFalse(np.any(cleaned[:, :10]))
        self.assertEqual(details["component_policy"], "bbox_center_or_nearest")


if __name__ == "__main__":
    unittest.main()
