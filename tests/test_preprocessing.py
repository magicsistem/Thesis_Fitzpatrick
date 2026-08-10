from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis_fitzpatrick.benchmark import ImageInput  # noqa: E402
from thesis_fitzpatrick.preprocessing import (  # noqa: E402
    BBox,
    CoordinateTransform,
    Detection,
    detect_and_inpaint_hair,
    detect_fov,
    detector_from_config,
    expand_bbox,
    run_p0,
)


CONFIG = json.loads((REPO_ROOT / "configs" / "benchmark" / "default.json").read_text())


class CountingDetector:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def identity(self) -> dict:
        return {"type": "synthetic", "version": 1}

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        self.calls += 1
        height, width = rgb.shape[:2]
        return [Detection(BBox(width // 4, height // 4, 3 * width // 4, 3 * height // 4), 0.9)]


class FOVTests(unittest.TestCase):
    def test_uniform_clear_and_dark_images_remain_full(self) -> None:
        for value in (210, 25):
            rgb = np.full((100, 140, 3), value, dtype=np.uint8)
            result = detect_fov(rgb, CONFIG["p0"]["fov"])
            self.assertTrue(np.all(result.mask == 1))
            self.assertTrue(result.fallback_full_image)

    def test_black_rectangular_frame_is_removed(self) -> None:
        rgb = np.full((120, 160, 3), 145, dtype=np.uint8)
        rgb[:12] = rgb[-12:] = 0
        rgb[:, :12] = rgb[:, -12:] = 0
        result = detect_fov(rgb, CONFIG["p0"]["fov"])
        self.assertFalse(result.fallback_full_image)
        self.assertEqual(int(result.mask[0, 0]), 0)
        self.assertEqual(int(result.mask[60, 80]), 1)
        self.assertGreater(result.valid_fraction, 0.65)

    def test_elliptical_fov_keeps_interior(self) -> None:
        rgb = np.zeros((160, 220, 3), dtype=np.uint8)
        cv2.ellipse(rgb, (110, 80), (96, 65), 0, 0, 360, (125, 95, 80), -1)
        result = detect_fov(rgb, CONFIG["p0"]["fov"])
        self.assertEqual(int(result.mask[80, 110]), 1)
        self.assertEqual(int(result.mask[0, 0]), 0)
        self.assertGreater(result.valid_fraction, 0.45)

    def test_disconnected_dark_interior_object_stays_valid(self) -> None:
        rgb = np.full((120, 160, 3), 150, dtype=np.uint8)
        rgb[:8] = rgb[-8:] = 0
        rgb[:, :8] = rgb[:, -8:] = 0
        rgb[55:65, 75:85] = 0
        result = detect_fov(rgb, CONFIG["p0"]["fov"])
        self.assertEqual(int(result.mask[60, 80]), 1)

    def test_scaled_behavior_across_resolutions(self) -> None:
        fractions = []
        for height, width in ((80, 120), (240, 360)):
            rgb = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.ellipse(rgb, (width // 2, height // 2), (int(width * 0.43), int(height * 0.42)), 0, 0, 360, (140, 100, 80), -1)
            fractions.append(detect_fov(rgb, CONFIG["p0"]["fov"]).valid_fraction)
        self.assertLess(abs(fractions[0] - fractions[1]), 0.08)


class HairTests(unittest.TestCase):
    def test_multiorientation_lines_are_detected_without_mutating_original(self) -> None:
        rgb = np.full((180, 220, 3), 165, dtype=np.uint8)
        cv2.line(rgb, (20, 30), (200, 80), (20, 20, 20), 2)
        cv2.line(rgb, (70, 10), (90, 170), (25, 25, 25), 2)
        original = rgb.copy()
        result = detect_and_inpaint_hair(rgb, np.ones(rgb.shape[:2], np.uint8), CONFIG["p0"]["hair"])
        self.assertGreater(np.count_nonzero(result.mask), 100)
        np.testing.assert_array_equal(rgb, original)
        self.assertFalse(np.array_equal(result.segmentation_input, original))

    def test_round_dark_lesion_is_not_entirely_labelled_as_hair(self) -> None:
        rgb = np.full((180, 180, 3), 170, dtype=np.uint8)
        cv2.circle(rgb, (90, 90), 35, (45, 45, 45), -1)
        result = detect_and_inpaint_hair(rgb, np.ones((180, 180), np.uint8), CONFIG["p0"]["hair"])
        lesion = np.zeros((180, 180), np.uint8)
        cv2.circle(lesion, (90, 90), 35, 1, -1)
        self.assertLess(np.mean(result.mask[lesion > 0]), 0.2)

    def test_abnormal_coverage_uses_explicit_skip_policy(self) -> None:
        rgb = np.full((120, 160, 3), 170, dtype=np.uint8)
        for y in range(10, 110, 8):
            cv2.line(rgb, (5, y), (155, y), (10, 10, 10), 2)
        config = copy.deepcopy(CONFIG["p0"]["hair"])
        config["maximum_coverage_fraction"] = 0.01
        result = detect_and_inpaint_hair(rgb, np.ones((120, 160), np.uint8), config)
        self.assertTrue(result.fallback_used)
        self.assertFalse(np.any(result.mask))
        np.testing.assert_array_equal(result.segmentation_input, rgb)


class CoordinateAndP0Tests(unittest.TestCase):
    def test_default_detector_is_explicit_missing_weights_fallback(self) -> None:
        detector = detector_from_config(CONFIG, REPO_ROOT)
        self.assertEqual(detector.identity["status"], "weights_missing")
        broken = copy.deepcopy(CONFIG)
        broken["p0"]["yolo"]["cfg_path"] = "models/yolo/model.cfg"
        with self.assertRaisesRegex(ValueError, "both cfg_path and weights_path"):
            detector_from_config(broken, REPO_ROOT)

    def test_border_boxes_and_margin_are_clipped(self) -> None:
        self.assertEqual(expand_bbox(BBox(0, 0, 20, 30), 0.3, 100, 80), BBox(0, 0, 26, 39))
        self.assertEqual(expand_bbox(BBox(0, 0, 100, 80), 0.3, 100, 80), BBox(0, 0, 100, 80))
        self.assertEqual(expand_bbox(BBox(80, 60, 100, 80), 0.3, 100, 80), BBox(74, 54, 100, 80))

    def test_binary_roi_round_trip_uses_nearest_neighbor(self) -> None:
        transform = CoordinateTransform(100, 80, BBox(10, 20, 70, 60))
        roi = np.zeros((7, 11), np.uint8)
        roi[2:5, 3:8] = 1
        original = transform.roi_to_original_mask(roi)
        self.assertEqual(set(np.unique(original)), {0, 1})
        restored = transform.original_to_roi_mask(original, size=(11, 7))
        self.assertEqual(set(np.unique(restored)), {0, 1})
        expected_y, expected_x = np.where(roi > 0)
        restored_y, restored_x = np.where(restored > 0)
        for expected, actual in zip(
            (expected_x.min(), expected_y.min(), expected_x.max(), expected_y.max()),
            (restored_x.min(), restored_y.min(), restored_x.max(), restored_y.max()),
        ):
            self.assertLessEqual(abs(int(expected) - int(actual)), 1)

    def test_p0_cache_runs_detector_once_and_preserves_complete_manifest(self) -> None:
        rgb = np.full((96, 128, 3), 140, dtype=np.uint8)
        image = ImageInput("one", "synthetic", "validation", rgb)
        detector = CountingDetector()
        with tempfile.TemporaryDirectory() as directory:
            first = run_p0(image, CONFIG, detector, cache_root=Path(directory))
            second = run_p0(image, CONFIG, detector, cache_root=Path(directory))
            self.assertEqual(detector.calls, 1)
            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertEqual(first.cache_key, second.cache_key)
            self.assertTrue((first.cache_directory / "preprocessing_manifest.json").is_file())
            self.assertEqual(first.roi_input.shape, second.roi_input.shape)
            self.assertIn("parameters", second.stage_details["fov"])
            self.assertIn("detector_identity", second.stage_details["yolo"])

    def test_missing_yolo_uses_full_fov_fallback(self) -> None:
        rgb = np.full((80, 100, 3), 120, dtype=np.uint8)
        result = run_p0(ImageInput("one", "synthetic", "validation", rgb), CONFIG)
        self.assertTrue(result.detector_failed)
        self.assertTrue(result.fallback_used)
        self.assertIsNone(result.selected_bbox_original)
        self.assertEqual(result.expanded_bbox_original, BBox(0, 0, 100, 80))

    def test_ablation_stages_disable_hair_and_yolo_without_calling_detector(self) -> None:
        rgb = np.full((80, 100, 3), 120, dtype=np.uint8)
        detector = CountingDetector()
        result = run_p0(
            ImageInput("one", "synthetic", "validation", rgb), CONFIG, detector,
            enable_fov=True, enable_hair=False, enable_yolo=False,
        )
        self.assertEqual(detector.calls, 0)
        self.assertFalse(np.any(result.hair_mask))
        self.assertFalse(result.detector_failed)
        self.assertFalse(result.fallback_used)
        self.assertEqual(result.stages, {"fov": True, "hair": False, "yolo": False})


if __name__ == "__main__":
    unittest.main()
