"""Binary segmentation metrics with explicit empty-case conventions."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def _safe_ratio(numerator: int | float, denominator: int | float, *, empty_value: float | None) -> tuple[float | None, bool]:
    return (empty_value, True) if denominator == 0 else (float(numerator / denominator), False)


def _boundary(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    return binary & (1 - cv2.erode(binary, np.ones((3, 3), np.uint8)))


def _boundary_metrics(prediction: np.ndarray, ground_truth: np.ndarray, tolerance: float) -> tuple[float, float | None, float | None, dict[str, bool]]:
    pred_boundary = _boundary(prediction)
    gt_boundary = _boundary(ground_truth)
    pred_count, gt_count = int(pred_boundary.sum()), int(gt_boundary.sum())
    flags = {"boundary_both_empty": pred_count == 0 and gt_count == 0, "hd95_undefined": False}
    if pred_count == 0 and gt_count == 0:
        return 1.0, 0.0, 0.0, flags
    if pred_count == 0 or gt_count == 0:
        flags["hd95_undefined"] = True
        return 0.0, None, None, flags
    distance_to_gt = cv2.distanceTransform((gt_boundary == 0).astype(np.uint8), cv2.DIST_L2, 5)
    distance_to_pred = cv2.distanceTransform((pred_boundary == 0).astype(np.uint8), cv2.DIST_L2, 5)
    pred_distances = distance_to_gt[pred_boundary > 0]
    gt_distances = distance_to_pred[gt_boundary > 0]
    precision = float(np.mean(pred_distances <= tolerance))
    recall = float(np.mean(gt_distances <= tolerance))
    boundary_f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    hd95 = float(np.percentile(np.concatenate((pred_distances, gt_distances)), 95))
    diagonal = math.hypot(*prediction.shape)
    return boundary_f1, hd95, hd95 / diagonal, flags


def segmentation_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    *,
    fov_mask: np.ndarray | None = None,
    hair_mask: np.ndarray | None = None,
    threshold_jaccard_cutoff: float = 0.65,
    boundary_tolerance_diagonal_fraction: float = 0.01,
    nearly_complete_fraction: float = 0.98,
) -> dict[str, Any]:
    """Compute binary segmentation metrics with explicit degenerate conventions.

    Empty prediction versus non-empty ground truth is a valid poor prediction:
    Jaccard, Dice, sensitivity and boundary F1 are zero. HD95 is undefined
    because one boundary is absent and is returned as ``None`` with the
    corresponding flag. The observation must not be silently discarded.
    """
    if prediction.shape != ground_truth.shape or prediction.ndim != 2:
        raise ValueError("prediction and ground truth must be same-size 2D masks")
    invalid_values = bool(not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(ground_truth)))
    pred, gt = np.nan_to_num(prediction) > 0, np.nan_to_num(ground_truth) > 0
    component_count = max(0, cv2.connectedComponents(pred.astype(np.uint8), connectivity=8)[0] - 1)
    prediction_touches_border = bool(np.any(pred[0]) or np.any(pred[-1]) or np.any(pred[:, 0]) or np.any(pred[:, -1]))
    ground_truth_touches_border = bool(np.any(gt[0]) or np.any(gt[-1]) or np.any(gt[:, 0]) or np.any(gt[:, -1]))
    true_positive = int(np.count_nonzero(pred & gt))
    false_positive = int(np.count_nonzero(pred & ~gt))
    false_negative = int(np.count_nonzero(~pred & gt))
    true_negative = int(np.count_nonzero(~pred & ~gt))
    union = true_positive + false_positive + false_negative
    both_empty = not np.any(pred) and not np.any(gt)
    jaccard = 1.0 if both_empty else float(true_positive / union)
    dice_denominator = 2 * true_positive + false_positive + false_negative
    dice = 1.0 if dice_denominator == 0 else float(2 * true_positive / dice_denominator)
    sensitivity, sensitivity_zero = _safe_ratio(true_positive, true_positive + false_negative, empty_value=1.0)
    specificity, specificity_zero = _safe_ratio(true_negative, true_negative + false_positive, empty_value=None)
    precision_empty = 1.0 if both_empty else 0.0
    precision, precision_zero = _safe_ratio(true_positive, true_positive + false_positive, empty_value=precision_empty)
    accuracy = float((true_positive + true_negative) / pred.size)
    diagonal = math.hypot(*pred.shape)
    tolerance = max(1.0, diagonal * boundary_tolerance_diagonal_fraction)
    boundary_f1, hd95, hd95_normalized, boundary_flags = _boundary_metrics(pred, gt, tolerance)

    fov_leak = None
    fov_leak_zero = False
    if fov_mask is not None:
        if fov_mask.shape != pred.shape:
            raise ValueError("FOV mask dimensions differ")
        fov_leak, fov_leak_zero = _safe_ratio(np.count_nonzero(pred & ~(fov_mask > 0)), np.count_nonzero(pred), empty_value=0.0)

    contamination = None
    contamination_zero = False
    if fov_mask is not None:
        hair = np.zeros_like(pred) if hair_mask is None else hair_mask > 0
        if hair.shape != pred.shape:
            raise ValueError("hair mask dimensions differ")
        clean_skin = (fov_mask > 0) & ~pred & ~hair
        contamination, contamination_zero = _safe_ratio(np.count_nonzero(clean_skin & gt), np.count_nonzero(clean_skin), empty_value=None)

    return {
        "threshold_jaccard": jaccard if jaccard >= threshold_jaccard_cutoff else 0.0,
        "jaccard": jaccard,
        "dice": dice,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "accuracy": accuracy,
        "boundary_f1": boundary_f1,
        "boundary_tolerance_pixels": tolerance,
        "hd95_pixels": hd95,
        "hd95_normalized": hd95_normalized,
        "fov_leak": fov_leak,
        "clean_skin_contamination": contamination,
        "counts": {"tp": true_positive, "fp": false_positive, "fn": false_negative, "tn": true_negative},
        "flags": {
            "ground_truth_empty": not np.any(gt),
            "prediction_empty": not np.any(pred),
            "both_empty": both_empty,
            "prediction_nearly_complete": float(np.mean(pred)) >= nearly_complete_fraction,
            "multiple_components": component_count > 1,
            "component_count": component_count,
            "prediction_touches_image_border": prediction_touches_border,
            "ground_truth_touches_image_border": ground_truth_touches_border,
            "nan_or_infinity": invalid_values,
            "sensitivity_denominator_zero": sensitivity_zero,
            "specificity_denominator_zero": specificity_zero,
            "precision_denominator_zero": precision_zero,
            "fov_leak_denominator_zero": fov_leak_zero,
            "contamination_denominator_zero": contamination_zero,
            **boundary_flags,
        },
    }


def paired_bootstrap(
    first: np.ndarray,
    second: np.ndarray,
    *,
    repetitions: int = 10_000,
    seed: int = 20260806,
) -> dict[str, float]:
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if first.shape != second.shape or first.ndim != 1 or len(first) == 0:
        raise ValueError("paired samples must be non-empty one-dimensional arrays of equal size")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("paired samples contain NaN or infinity")
    differences = first - second
    generator = np.random.default_rng(seed)
    samples = differences[generator.integers(0, len(differences), size=(repetitions, len(differences)))].mean(axis=1)
    low, high = np.percentile(samples, (2.5, 97.5))
    return {"mean_difference": float(differences.mean()), "ci95_low": float(low), "ci95_high": float(high), "repetitions": repetitions, "seed": seed}


def paired_permutation_pvalue(first: np.ndarray, second: np.ndarray, *, repetitions: int = 10_000, seed: int = 20260806) -> float:
    differences = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    if differences.ndim != 1 or len(differences) == 0 or not np.all(np.isfinite(differences)):
        raise ValueError("paired finite samples are required")
    observed = abs(float(differences.mean()))
    generator = np.random.default_rng(seed)
    signs = generator.choice((-1.0, 1.0), size=(repetitions, len(differences)))
    simulated = np.abs((signs * differences).mean(axis=1))
    return float((np.count_nonzero(simulated >= observed) + 1) / (repetitions + 1))


def mcnemar_exact(success_first: np.ndarray, success_second: np.ndarray) -> dict[str, float | int]:
    first, second = np.asarray(success_first, dtype=bool), np.asarray(success_second, dtype=bool)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("paired one-dimensional success arrays are required")
    first_only = int(np.count_nonzero(first & ~second))
    second_only = int(np.count_nonzero(~first & second))
    discordant = first_only + second_only
    if discordant == 0:
        pvalue = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(0, min(first_only, second_only) + 1)) / (2 ** discordant)
        pvalue = min(1.0, 2 * tail)
    return {"first_only": first_only, "second_only": second_only, "discordant": discordant, "pvalue": pvalue}


def holm_adjust(pvalues: list[float]) -> list[float]:
    if any(not 0 <= value <= 1 for value in pvalues):
        raise ValueError("p-values must be in [0, 1]")
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(sorted(range(len(pvalues)), key=pvalues.__getitem__)):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted
