from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis_fitzpatrick.benchmark import ImageInput  # noqa: E402
import thesis_fitzpatrick.preprocessing as preprocessing  # noqa: E402
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
from scripts.benchmark import prepare_p0_oof as p0_oof  # noqa: E402


CONFIG = json.loads((REPO_ROOT / "configs" / "benchmark" / "default.json").read_text())


def _reference_component_points(labels: np.ndarray, _stats: np.ndarray, label: int) -> np.ndarray | None:
    points = np.column_stack(np.where(labels == label))[:, ::-1].astype(np.float32)
    return points if len(points) >= 3 else None


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
    def test_component_scan_matches_full_image_reference(self) -> None:
        candidate = np.zeros((96, 128), np.uint8)
        candidate[0, :7] = 1
        candidate[-1, -8:] = 1
        cv2.line(candidate, (4, 60), (80, 60), 1, 1)
        cv2.line(candidate, (100, 3), (100, 82), 1, 1)
        cv2.line(candidate, (12, 88), (88, 12), 1, 1)
        rng = np.random.default_rng(73)
        noise_y, noise_x = np.where(rng.random(candidate.shape) < 0.02)
        candidate[noise_y, noise_x] = 1
        count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
        for label in range(1, count):
            expected = _reference_component_points(labels, stats, label)
            actual = preprocessing._component_points(labels, stats, label)
            if expected is None:
                self.assertIsNone(actual)
                continue
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(cv2.minAreaRect(actual), cv2.minAreaRect(expected))

    def test_optimized_hair_result_matches_reference_component_scan(self) -> None:
        rgb = np.full((160, 200, 3), 165, dtype=np.uint8)
        cv2.line(rgb, (0, 2), (199, 2), (20, 20, 20), 2)
        cv2.line(rgb, (9, 0), (9, 159), (20, 20, 20), 2)
        cv2.line(rgb, (15, 145), (180, 20), (20, 20, 20), 2)
        rng = np.random.default_rng(19)
        noise = rng.random(rgb.shape[:2]) < 0.01
        rgb[noise] = (50, 50, 50)
        fov = np.ones(rgb.shape[:2], np.uint8)
        with patch.object(preprocessing, "_component_points", _reference_component_points):
            expected = detect_and_inpaint_hair(rgb, fov, CONFIG["p0"]["hair"])
        actual = detect_and_inpaint_hair(rgb, fov, CONFIG["p0"]["hair"])
        np.testing.assert_array_equal(actual.mask, expected.mask)
        np.testing.assert_array_equal(actual.segmentation_input, expected.segmentation_input)
        self.assertEqual(actual.coverage_fraction, expected.coverage_fraction)
        self.assertEqual(actual.fallback_used, expected.fallback_used)
        self.assertEqual(actual.warnings, expected.warnings)
        self.assertEqual(actual.parameters, expected.parameters)

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
            (first.cache_directory / "yolo_overlay.png").unlink()
            rebuilt = run_p0(image, CONFIG, detector, cache_root=Path(directory))
            self.assertFalse(rebuilt.cache_hit)
            self.assertEqual(detector.calls, 2)
            (rebuilt.cache_directory / "yolo_bbox.json").write_text("not json", encoding="utf-8")
            repaired = run_p0(image, CONFIG, detector, cache_root=Path(directory))
            self.assertFalse(repaired.cache_hit)
            self.assertEqual(detector.calls, 3)

    def test_oof_workers_validate_structure_and_process_each_image_once(self) -> None:
        self.assertEqual(p0_oof.resolve_workers(None, 32), 24)
        with self.assertRaisesRegex(ValueError, "1 y 32"):
            p0_oof.resolve_workers(33, 32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); data = root / "data"; data.mkdir(); yolo = root / "yolo"
            items = []
            for index in range(5):
                image_id = f"image-{index}"
                Image.fromarray(np.full((40, 60, 3), 120 + index, dtype=np.uint8)).save(data / f"{image_id}.png")
                items.append({"image_id": image_id, "image_path": f"{image_id}.png"})
                frozen = yolo / f"fold-{index}"; frozen.mkdir(parents=True)
                (frozen / "frozen.json").write_text(json.dumps({"status": "frozen", "fold": index}), encoding="utf-8")
            manifest = {"items": items}
            folds = {"folds": [{"fold": index, "validation_ids": [f"image-{index}"]} for index in range(5)]}
            by_id, plan = p0_oof.validate_oof_structure(manifest, folds, yolo)
            self.assertEqual({fold for fold, _ in plan}, set(range(5)))
            self.assertEqual(p0_oof.fold_config(CONFIG, yolo, 3)["p0"]["yolo"]["frozen_manifest"], str((yolo / "fold-3" / "frozen.json").resolve()))
            config = copy.deepcopy(CONFIG)
            ids = [f"image-{index}" for index in range(5)]
            outcome = p0_oof.process_fold(0, ids, by_id, config, data_root=data, artifact_root=root / "parallel", dataset_id="synthetic", workers=2, reuse_cache=True)
            sequential_detector = detector_from_config(config, REPO_ROOT)
            with Image.open(data / "image-0.png") as opened:
                sequential = run_p0(ImageInput("image-0", "synthetic", "train_oof", np.asarray(opened.convert("RGB"), dtype=np.uint8)), config, sequential_detector, cache_root=root / "sequential", reuse_cache=True)
            self.assertEqual({item["image_id"] for item in outcome}, set(ids))
            self.assertTrue(all("error" not in item and not item["cache_hit"] for item in outcome))
            self.assertEqual(next(item for item in outcome if item["image_id"] == "image-0")["cache_key"], sequential.cache_key)
            repeated = p0_oof.process_fold(0, ids, by_id, config, data_root=data, artifact_root=root / "parallel", dataset_id="synthetic", workers=2, reuse_cache=True)
            self.assertTrue(all(item["cache_hit"] for item in repeated))
            failed = p0_oof.process_fold(0, ["missing"], {**by_id, "missing": {"image_id": "missing", "image_path": "missing.png"}}, config, data_root=data, artifact_root=root / "parallel", dataset_id="synthetic", workers=1, reuse_cache=True)
            self.assertIn("error", failed[0])
            bad = {"folds": [{"fold": 0, "validation_ids": ["image-0", "image-1"]}, *[{"fold": index, "validation_ids": [f"image-{index}"]} for index in range(1, 5)] ]}
            with self.assertRaisesRegex(ValueError, "duplicate"):
                p0_oof.validate_oof_structure(manifest, bad, yolo)
            self.assertEqual(p0_oof.completion_status(5, 5, []), "completed")
            self.assertEqual(p0_oof.completion_status(4, 5, []), "failed")
            self.assertEqual(p0_oof.completion_status(5, 5, [{"error": "x"}]), "failed")

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
