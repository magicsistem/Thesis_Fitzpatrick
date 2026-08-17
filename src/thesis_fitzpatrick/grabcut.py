"""S16 GrabCut backend and shared post-processing."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from .benchmark import BackendResult
from .preprocessing import BBox, P0Result, bbox_from_mask


VALID_GRABCUT_LABELS = {cv2.GC_BGD, cv2.GC_FGD, cv2.GC_PR_BGD, cv2.GC_PR_FGD}


def classify_mask_failure(mask: np.ndarray, nearly_complete_fraction: float = 0.98) -> tuple[str | None, list[str]]:
    """Classify GrabCut output diagnostics.

    Empty/nearly-complete outputs are diagnostic quality codes, not execution
    failures. The benchmark decides separately whether a code is fatal.
    """
    fraction = float(np.mean(mask > 0))
    if not np.all(np.isfinite(mask)):
        return "nan_or_inf", ["GrabCut produced NaN or infinity."]
    if fraction == 0:
        return "empty_mask", ["GrabCut produced an empty mask."]
    if fraction >= nearly_complete_fraction:
        return "nearly_complete_mask", [f"GrabCut foreground fraction is {fraction:.4f}."]
    return None, []


def _prepare_image(rgb: np.ndarray, color_space: str) -> np.ndarray:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("GrabCut input must be HxWx3 uint8 RGB")
    if color_space == "bgr":
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if color_space == "lab":
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    raise ValueError("GrabCut color_space must be 'bgr' or 'lab'")


def grabcut_classic(
    rgb: np.ndarray,
    *,
    margin_fraction: float = 0.05,
    iterations: int = 5,
    nearly_complete_fraction: float = 0.98,
) -> BackendResult:
    """S16-Classic: original image and reproducible central rectangle."""
    if not 0 < margin_fraction < 0.5 or iterations < 1:
        raise ValueError("invalid GrabCut classic parameters")
    height, width = rgb.shape[:2]
    margin_x = max(1, int(round(width * margin_fraction)))
    margin_y = max(1, int(round(height * margin_fraction)))
    rect = (margin_x, margin_y, width - 2 * margin_x, height - 2 * margin_y)
    labels = np.zeros((height, width), dtype=np.uint8)
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    started = time.perf_counter()
    failure_code = None
    warnings: list[str] = []
    try:
        cv2.grabCut(
            _prepare_image(rgb, "bgr"), labels, rect,
            background_model, foreground_model, iterations, cv2.GC_INIT_WITH_RECT,
        )
        mask = np.isin(labels, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    except cv2.error as exc:
        mask = np.zeros((height, width), dtype=np.uint8)
        failure_code = "grabcut_error"
        warnings.append(str(exc))
    elapsed = (time.perf_counter() - started) * 1000
    if failure_code is None:
        failure_code, mask_warnings = classify_mask_failure(mask, nearly_complete_fraction)
        warnings.extend(mask_warnings)
    x, y, w, h = rect
    touches_rectangle = bool(
        np.any(mask[y, x:x + w])
        or np.any(mask[y + h - 1, x:x + w])
        or np.any(mask[y:y + h, x])
        or np.any(mask[y:y + h, x + w - 1])
    )
    if touches_rectangle:
        warnings.append("La predicción toca el rectángulo de inicialización y podría estar cortada.")
    return BackendResult(
        native_mask=mask,
        input_size=(width, height),
        backend_time_ms=elapsed,
        model_identity={"method_id": "S16", "backend": "GrabCut", "mode": "classic", "opencv": cv2.__version__},
        warnings=warnings,
        failure_code=failure_code,
        details={
            "initialization": "GC_INIT_WITH_RECT",
            "margin_fraction": margin_fraction,
            "iterations": iterations,
            "color_space": "bgr",
            "rect_xywh": list(rect),
            "prediction_touches_initialization_rect": touches_rectangle,
            "ground_truth_used": False,
        },
    )


def _selected_box_in_roi(p0: P0Result) -> BBox | None:
    selected = p0.selected_bbox_original
    roi = p0.expanded_bbox_original
    if selected is None:
        return None
    return BBox(
        max(0, selected.x0 - roi.x0),
        max(0, selected.y0 - roi.y0),
        min(roi.width, selected.x1 - roi.x0),
        min(roi.height, selected.y1 - roi.y0),
    )


def build_robust_trimap(p0: P0Result, fallback_margin_fraction: float = 0.05) -> np.ndarray:
    """Build an OpenCV trimap without ground truth or another model's mask."""
    fov = p0.roi_fov_mask > 0
    height, width = fov.shape
    trimap = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
    selected = _selected_box_in_roi(p0)
    if selected is not None:
        trimap[fov] = cv2.GC_PR_BGD
        region = np.zeros_like(fov)
        region[selected.y0:selected.y1, selected.x0:selected.x1] = True
        trimap[region & fov] = cv2.GC_PR_FGD
        # Pixels beyond the expanded box are not part of the ROI. Its boundary
        # represents that sure-background exterior for GrabCut.
        trimap[0] = trimap[-1] = cv2.GC_BGD
        trimap[:, 0] = trimap[:, -1] = cv2.GC_BGD
    else:
        valid_box = bbox_from_mask(p0.roi_fov_mask)
        margin_x = max(1, int(round(valid_box.width * fallback_margin_fraction)))
        margin_y = max(1, int(round(valid_box.height * fallback_margin_fraction)))
        x0, y0 = valid_box.x0 + margin_x, valid_box.y0 + margin_y
        x1, y1 = valid_box.x1 - margin_x, valid_box.y1 - margin_y
        if x1 <= x0 or y1 <= y0:
            raise ValueError("FOV is too small for the GrabCut fallback rectangle")
        inner = np.zeros_like(fov)
        inner[y0:y1, x0:x1] = True
        trimap[inner & fov] = cv2.GC_PR_FGD
    return trimap


def grabcut_robust(
    p0: P0Result,
    *,
    iterations: int = 5,
    color_space: str = "bgr",
    fallback_margin_fraction: float = 0.05,
    nearly_complete_fraction: float = 0.98,
) -> BackendResult:
    """S16-Robust over the same P0 ROI consumed by neural backends."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    labels = build_robust_trimap(p0, fallback_margin_fraction)
    if not set(np.unique(labels)).issubset(VALID_GRABCUT_LABELS):
        raise RuntimeError("invalid GrabCut trimap labels")
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    started = time.perf_counter()
    failure_code = None
    warnings: list[str] = []
    try:
        cv2.grabCut(
            _prepare_image(p0.roi_input, color_space), labels, None,
            background_model, foreground_model, iterations, cv2.GC_INIT_WITH_MASK,
        )
        mask = np.isin(labels, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    except cv2.error as exc:
        mask = np.zeros(p0.roi_input.shape[:2], dtype=np.uint8)
        failure_code = "grabcut_error"
        warnings.append(str(exc))
    elapsed = (time.perf_counter() - started) * 1000
    if failure_code is None:
        failure_code, mask_warnings = classify_mask_failure(mask, nearly_complete_fraction)
        warnings.extend(mask_warnings)
    return BackendResult(
        native_mask=mask,
        input_size=(p0.roi_input.shape[1], p0.roi_input.shape[0]),
        backend_time_ms=elapsed,
        model_identity={"method_id": "S16", "backend": "GrabCut", "mode": "robust", "opencv": cv2.__version__},
        warnings=warnings,
        failure_code=failure_code,
        details={
            "initialization": "GC_INIT_WITH_MASK",
            "iterations": iterations,
            "color_space": color_space,
            "fallback_margin_fraction": fallback_margin_fraction if p0.selected_bbox_original is None else None,
            "trimap_labels": sorted(int(value) for value in np.unique(labels)),
            "detector_fallback": p0.fallback_used,
            "ground_truth_used": False,
        },
    )


def _scaled_odd(minimum_dimension: int, fraction: float) -> int:
    size = max(3, int(round(minimum_dimension * fraction)))
    return size if size % 2 else size + 1


def common_postprocess(
    restored_mask: np.ndarray,
    fov_mask: np.ndarray,
    config: dict[str, Any],
    *,
    selected_bbox: BBox | None = None,
    apply_fov: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the same proportional cleanup after every common-pipeline backend."""
    if restored_mask.shape != fov_mask.shape:
        raise ValueError("restored mask and FOV dimensions differ")
    mask = (restored_mask > 0).astype(np.uint8)
    if apply_fov:
        mask[fov_mask == 0] = 0
    minimum_dimension = min(mask.shape)
    opening = _scaled_odd(minimum_dimension, float(config["opening_kernel_fraction"]))
    closing = _scaled_odd(minimum_dimension, float(config["closing_kernel_fraction"]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((opening, opening), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((closing, closing), np.uint8))

    maximum_hole_area = int(round(mask.size * float(config["maximum_hole_area_fraction"])))
    inverse = (mask == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
    border_labels = set(np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))))
    holes_filled = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if label not in border_labels and area <= maximum_hole_area:
            mask[labels == label] = 1
            holes_filled += 1

    minimum_area = int(round(mask.size * float(config["minimum_component_area_fraction"])))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = [label for label in range(1, count) if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area]
    components_before_selection = len(candidates)
    chosen = None
    if candidates:
        if selected_bbox is not None:
            target = np.array([(selected_bbox.x0 + selected_bbox.x1 - 1) / 2, (selected_bbox.y0 + selected_bbox.y1 - 1) / 2])
            center_x, center_y = int(round(target[0])), int(round(target[1]))
            if 0 <= center_y < mask.shape[0] and 0 <= center_x < mask.shape[1] and labels[center_y, center_x] in candidates:
                chosen = int(labels[center_y, center_x])
            else:
                chosen = min(candidates, key=lambda label: float(np.linalg.norm(centroids[label] - target)))
        else:
            chosen = max(candidates, key=lambda label: int(stats[label, cv2.CC_STAT_AREA]))
    cleaned = (labels == chosen).astype(np.uint8) if chosen is not None else np.zeros_like(mask)
    return cleaned, {
        "opening_kernel": opening,
        "closing_kernel": closing,
        "holes_filled": holes_filled,
        "minimum_component_area_pixels": minimum_area,
        "components_before_selection": components_before_selection,
        "component_policy": "bbox_center_or_nearest" if selected_bbox is not None else "largest_component",
        "fov_intersection_applied": apply_fov,
    }
