from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis_fitzpatrick.metrics import (  # noqa: E402
    holm_adjust,
    mcnemar_exact,
    paired_bootstrap,
    paired_permutation_pvalue,
    segmentation_metrics,
)


class MetricTests(unittest.TestCase):
    def test_perfect_prediction(self) -> None:
        mask = np.zeros((20, 20), np.uint8)
        mask[5:15, 6:14] = 1
        metrics = segmentation_metrics(mask, mask, fov_mask=np.ones_like(mask), hair_mask=np.zeros_like(mask))
        for name in ("threshold_jaccard", "jaccard", "dice", "sensitivity", "specificity", "precision", "accuracy", "boundary_f1"):
            self.assertEqual(metrics[name], 1.0)
        self.assertEqual(metrics["hd95_pixels"], 0.0)

    def test_no_intersection(self) -> None:
        pred, gt = np.zeros((20, 20), np.uint8), np.zeros((20, 20), np.uint8)
        pred[2:6, 2:6], gt[12:16, 12:16] = 1, 1
        metrics = segmentation_metrics(pred, gt)
        self.assertEqual(metrics["threshold_jaccard"], 0.0)
        self.assertEqual(metrics["jaccard"], 0.0)
        self.assertEqual(metrics["dice"], 0.0)
        self.assertGreater(metrics["hd95_pixels"], 0)

    def test_both_empty_convention(self) -> None:
        empty = np.zeros((10, 10), np.uint8)
        metrics = segmentation_metrics(empty, empty, fov_mask=np.ones_like(empty))
        self.assertEqual(metrics["jaccard"], 1.0)
        self.assertEqual(metrics["dice"], 1.0)
        self.assertEqual(metrics["boundary_f1"], 1.0)
        self.assertEqual(metrics["fov_leak"], 0.0)
        self.assertTrue(metrics["flags"]["both_empty"])
        self.assertTrue(metrics["flags"]["fov_leak_denominator_zero"])

    def test_only_one_empty_has_typed_hd95(self) -> None:
        empty, nonempty = np.zeros((10, 10), np.uint8), np.zeros((10, 10), np.uint8)
        nonempty[3:7, 3:7] = 1
        for pred, gt in ((empty, nonempty), (nonempty, empty)):
            metrics = segmentation_metrics(pred, gt)
            self.assertEqual(metrics["jaccard"], 0.0)
            self.assertIsNone(metrics["hd95_pixels"])
            self.assertTrue(metrics["flags"]["hd95_undefined"])

    def test_fov_leak_and_clean_skin_zero_denominators(self) -> None:
        pred = np.ones((5, 5), np.uint8)
        gt = np.ones((5, 5), np.uint8)
        fov = np.ones((5, 5), np.uint8)
        metrics = segmentation_metrics(pred, gt, fov_mask=fov)
        self.assertIsNone(metrics["clean_skin_contamination"])
        self.assertTrue(metrics["flags"]["contamination_denominator_zero"])
        pred[:, :2] = 0
        fov[:, 3:] = 0
        metrics = segmentation_metrics(pred, gt, fov_mask=fov)
        self.assertGreater(metrics["fov_leak"], 0)

    def test_paired_statistics_are_deterministic_and_holm_monotone(self) -> None:
        first = np.array([0.8, 0.9, 0.7, 0.85])
        second = np.array([0.6, 0.7, 0.75, 0.65])
        self.assertEqual(paired_bootstrap(first, second, repetitions=100, seed=4), paired_bootstrap(first, second, repetitions=100, seed=4))
        self.assertEqual(paired_permutation_pvalue(first, second, repetitions=100, seed=4), paired_permutation_pvalue(first, second, repetitions=100, seed=4))
        self.assertEqual(mcnemar_exact([1, 1, 0], [0, 1, 1])["discordant"], 2)
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        self.assertTrue(all(0 <= value <= 1 for value in adjusted))


if __name__ == "__main__":
    unittest.main()
