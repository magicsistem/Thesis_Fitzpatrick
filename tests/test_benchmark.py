from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis_fitzpatrick.benchmark import (  # noqa: E402
    ImageInput,
    build_grouped_folds,
    evaluation_registry,
    load_benchmark_methods,
    validate_dataset_manifest,
    validate_dataset_registry,
)


class BenchmarkContractTests(unittest.TestCase):
    def test_registry_contains_real_fifteen_plus_s16(self) -> None:
        methods = load_benchmark_methods(REPO_ROOT / "configs" / "segmentation_models.json")
        self.assertEqual([item["method_id"] for item in methods], [f"S{i:02d}" for i in range(1, 17)])
        self.assertEqual(methods[-1]["backend_id"], "grabcut")
        self.assertEqual(sum(item["kind"] == "neural" for item in methods), 15)

    def test_b2_is_not_relabelled_b1_without_standardized_checkpoints(self) -> None:
        evaluations = {item["id"]: item for item in evaluation_registry(
            load_benchmark_methods(REPO_ROOT / "configs" / "segmentation_models.json")
        )}
        self.assertTrue(evaluations["B1"]["available"])
        self.assertFalse(evaluations["B2"]["available"])
        self.assertEqual(len(evaluations["B2"]["missing_methods"]), 15)

    def test_novice_masks_are_disabled_with_required_warning(self) -> None:
        payload = json.loads((REPO_ROOT / "configs" / "benchmark" / "datasets.json").read_text())
        datasets = validate_dataset_registry(payload)
        novice = next(item for item in datasets if item["id"] == "isic2018_novice_masks")
        self.assertFalse(novice["enabled"])
        self.assertIn("NO APTO", novice["warning"])

    def test_manifest_rejects_absolute_or_parent_paths(self) -> None:
        for image_path in ("/private/image.jpg", "../image.jpg"):
            payload = {
                "schema_version": 1,
                "dataset_id": "synthetic",
                "split": "train",
                "items": [{"image_id": "one", "image_path": image_path, "mask_paths": []}],
            }
            with self.assertRaisesRegex(ValueError, "relative POSIX"):
                validate_dataset_manifest(payload)

    def test_manifest_file_check_uses_explicit_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "images" / "one.jpg").write_bytes(b"image")
            payload = {
                "schema_version": 1,
                "dataset_id": "synthetic",
                "split": "train",
                "items": [{"image_id": "one", "image_path": "images/one.jpg", "mask_paths": []}],
            }
            audit = validate_dataset_manifest(payload, data_root=root, require_files=True)
            self.assertEqual(audit["items"], 1)

    def test_grouped_folds_keep_linked_identifiers_together(self) -> None:
        payload = {
            "schema_version": 1,
            "dataset_id": "synthetic",
            "split": "train",
            "items": [
                {"image_id": "a", "image_path": "a.jpg", "mask_paths": [], "patient_id": "p1", "duplicate_group_id": "d1"},
                {"image_id": "b", "image_path": "b.jpg", "mask_paths": [], "patient_id": "p1", "duplicate_group_id": "d2"},
                {"image_id": "c", "image_path": "c.jpg", "mask_paths": [], "patient_id": "p2", "duplicate_group_id": "d2"},
                {"image_id": "d", "image_path": "d.jpg", "mask_paths": [], "duplicate_group_id": "d"},
            ],
        }
        result = build_grouped_folds(payload, folds=2, seed=7)
        assigned = {
            image_id: fold["fold"]
            for fold in result["folds"]
            for image_id in fold["validation_ids"]
        }
        self.assertEqual(assigned["a"], assigned["b"])
        self.assertEqual(assigned["b"], assigned["c"])
        self.assertEqual(result["image_id_fallback_count"], 1)
        self.assertTrue(result["leakage_audit"]["passed"])

    def test_ground_truth_requires_explicit_permission(self) -> None:
        rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "cannot use"):
            ImageInput("one", "set", "test", rgb, ground_truth=np.zeros((4, 5), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
