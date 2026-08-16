"""Shared P0 preprocessing for all segmentation backends."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import time
from typing import Any, Protocol

import cv2
import numpy as np

from .benchmark import ImageInput, atomic_write_bytes, atomic_write_json, content_hash, load_json, sha256_file


@dataclass(frozen=True)
class BBox:
    """Integer ``xyxy`` box with an exclusive lower-right corner."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        if self.x0 < 0 or self.y0 < 0 or self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(f"invalid half-open bbox: {self}")

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def as_list(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]

    @classmethod
    def from_list(cls, values: list[int]) -> "BBox":
        return cls(*(int(value) for value in values))


@dataclass(frozen=True)
class CoordinateTransform:
    original_width: int
    original_height: int
    roi_bbox: BBox

    def roi_to_original_mask(self, mask: np.ndarray) -> np.ndarray:
        if mask.ndim != 2:
            raise ValueError("binary mask must be two-dimensional")
        resized = cv2.resize(
            (mask > 0).astype(np.uint8),
            (self.roi_bbox.width, self.roi_bbox.height),
            interpolation=cv2.INTER_NEAREST,
        )
        restored = np.zeros((self.original_height, self.original_width), dtype=np.uint8)
        restored[self.roi_bbox.y0:self.roi_bbox.y1, self.roi_bbox.x0:self.roi_bbox.x1] = resized
        return restored

    def original_to_roi_mask(self, mask: np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
        if mask.shape != (self.original_height, self.original_width):
            raise ValueError("mask dimensions do not match the original image")
        cropped = mask[self.roi_bbox.y0:self.roi_bbox.y1, self.roi_bbox.x0:self.roi_bbox.x1]
        if size is None or size == (self.roi_bbox.width, self.roi_bbox.height):
            return (cropped > 0).astype(np.uint8)
        return cv2.resize((cropped > 0).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)

    def manifest(self) -> dict[str, Any]:
        return {
            "bbox_convention": "xyxy_half_open",
            "original_size": [self.original_width, self.original_height],
            "roi_bbox_original": self.roi_bbox.as_list(),
            "roi_size": [self.roi_bbox.width, self.roi_bbox.height],
        }


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    confidence: float
    class_name: str = "lesion"

    def manifest(self) -> dict[str, Any]:
        return {"bbox_xyxy_original": self.bbox.as_list(), "confidence": self.confidence, "class_name": self.class_name}


class Detector(Protocol):
    @property
    def identity(self) -> dict[str, Any]: ...

    def detect(self, rgb: np.ndarray) -> list[Detection]: ...


class MissingYoloV3Detector:
    """Explicit no-weights detector used to exercise the documented fallback."""

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "architecture": "YOLOv3-Darknet53",
            "status": "weights_missing",
            "warning": "Configure local cfg_path and weights_path; no detector result was fabricated.",
        }

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        return []


class OpenCVYoloV3Detector:
    """One-class Darknet YOLOv3 inference using OpenCV DNN and local weights."""

    def __init__(
        self,
        cfg_path: Path,
        weights_path: Path,
        *,
        input_size: tuple[int, int] = (512, 512),
        confidence_threshold: float,
        nms_threshold: float,
        margin_fraction: float | None = None,
    ) -> None:
        missing = [str(path) for path in (cfg_path, weights_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError("YOLOv3 local resources are missing: " + ", ".join(missing))
        if not 0 < confidence_threshold < 1 or not 0 < nms_threshold < 1:
            raise ValueError("YOLO confidence and NMS thresholds must be frozen in (0, 1)")
        self.cfg_path = cfg_path
        self.weights_path = weights_path
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.margin_fraction = margin_fraction
        self.network = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))
        self.execution_device = "cpu"
        if os.environ.get("THESIS_OPENCV_DNN_DEVICE") == "cuda":
            if not hasattr(cv2, "cuda") or cv2.cuda.getCudaEnabledDeviceCount() < 1:
                raise RuntimeError("OpenCV DNN CUDA was requested but this OpenCV build has no visible CUDA device")
            self.network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            self.execution_device = "cuda"
        self._identity = {
            "architecture": "YOLOv3-Darknet53",
            "cfg_sha256": sha256_file(self.cfg_path),
            "weights_sha256": sha256_file(self.weights_path),
            "input_size": list(self.input_size),
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "margin_fraction": self.margin_fraction,
            "execution_device": self.execution_device,
        }

    @property
    def identity(self) -> dict[str, Any]:
        return self._identity

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        height, width = rgb.shape[:2]
        blob = cv2.dnn.blobFromImage(rgb, 1 / 255.0, self.input_size, swapRB=False, crop=False)
        self.network.setInput(blob)
        outputs = self.network.forward(self.network.getUnconnectedOutLayersNames())
        boxes: list[list[int]] = []
        confidences: list[float] = []
        for output in outputs:
            for row in output:
                if len(row) < 6:
                    continue
                confidence = float(row[4] * row[5])
                if confidence < self.confidence_threshold:
                    continue
                center_x, center_y = float(row[0] * width), float(row[1] * height)
                box_width, box_height = float(row[2] * width), float(row[3] * height)
                x = max(0, int(round(center_x - box_width / 2)))
                y = max(0, int(round(center_y - box_height / 2)))
                w = max(1, min(width - x, int(round(box_width))))
                h = max(1, min(height - y, int(round(box_height))))
                boxes.append([x, y, w, h])
                confidences.append(confidence)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        detections = []
        for index in np.asarray(indices).reshape(-1) if len(indices) else []:
            x, y, w, h = boxes[int(index)]
            detections.append(Detection(BBox(x, y, x + w, y + h), confidences[int(index)]))
        return sorted(detections, key=lambda detection: detection.confidence, reverse=True)


def detector_from_config(config: dict[str, Any], root: Path) -> Detector:
    """Create the configured local YOLOv3 detector or the explicit fallback."""
    yolo = config["p0"]["yolo"]
    frozen_value = yolo.get("frozen_manifest")
    frozen = None
    if frozen_value:
        frozen_path = Path(frozen_value)
        if not frozen_path.is_absolute(): frozen_path = root / frozen_path
        frozen = load_json(frozen_path)
        identity = frozen.pop("identity_hash", None)
        if frozen.get("schema_version") != 2 or frozen.get("status") != "frozen" or frozen.get("architecture") != "YOLOv3-Darknet53" or not isinstance(frozen.get("fold"), int) or content_hash(frozen) != identity:
            raise ValueError("El manifest YOLO no está congelado o no corresponde a YOLOv3-Darknet53")
        frozen["identity_hash"] = identity
        training_state = Path(frozen.get("training_state_path", ""))
        if not training_state.is_file() or sha256_file(training_state) != frozen.get("training_state_sha256"):
            raise ValueError("El estado de entrenamiento YOLO congelado falta o fue modificado")
        cfg_value, weights_value = frozen.get("cfg_path"), frozen.get("weights_path")
        thresholds = frozen.get("thresholds", {})
        yolo = {**yolo, "cfg_path": cfg_value, "weights_path": weights_value, "confidence_threshold": thresholds.get("confidence_threshold"), "nms_threshold": thresholds.get("nms_threshold")}
    cfg_value, weights_value = yolo.get("cfg_path"), yolo.get("weights_path")
    if cfg_value is None and weights_value is None:
        return MissingYoloV3Detector()
    if not cfg_value or not weights_value:
        raise ValueError("YOLOv3 requires both cfg_path and weights_path, or both null for the recorded FOV fallback")
    if yolo.get("confidence_threshold") is None or yolo.get("nms_threshold") is None:
        raise ValueError("YOLOv3 confidence_threshold and nms_threshold must be frozen before inference")
    cfg_path, weights_path = Path(cfg_value), Path(weights_value)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    if not weights_path.is_absolute():
        weights_path = root / weights_path
    if frozen:
        if sha256_file(cfg_path) != frozen.get("cfg_sha256") or sha256_file(weights_path) != frozen.get("weights_sha256"):
            raise ValueError("Los archivos YOLOv3 no coinciden con los hashes del manifest congelado")
    return OpenCVYoloV3Detector(
        cfg_path,
        weights_path,
        input_size=tuple(int(value) for value in yolo["input_size"]),
        confidence_threshold=float(yolo["confidence_threshold"]),
        nms_threshold=float(yolo["nms_threshold"]),
        margin_fraction=float(frozen["thresholds"]["margin_fraction"]) if frozen else None,
    )


@dataclass
class FOVResult:
    mask: np.ndarray = field(repr=False)
    fallback_full_image: bool
    valid_fraction: float
    confidence: str
    warnings: list[str]
    parameters: dict[str, Any]


@dataclass
class HairResult:
    mask: np.ndarray = field(repr=False)
    segmentation_input: np.ndarray = field(repr=False)
    coverage_fraction: float
    fallback_used: bool
    warnings: list[str]
    parameters: dict[str, Any]


@dataclass
class P0Result:
    image: ImageInput
    fov_mask: np.ndarray = field(repr=False)
    hair_mask: np.ndarray = field(repr=False)
    segmentation_input: np.ndarray = field(repr=False)
    detections: list[Detection]
    selected_bbox_original: BBox | None
    expanded_bbox_original: BBox
    roi_input: np.ndarray = field(repr=False)
    roi_fov_mask: np.ndarray = field(repr=False)
    transform: CoordinateTransform
    detector_failed: bool
    detector_status: str
    fallback_used: bool
    timings_ms: dict[str, float]
    warnings: list[str]
    configuration_hash: str
    cache_key: str
    stages: dict[str, bool]
    stage_details: dict[str, Any]
    cache_hit: bool = False
    cache_directory: Path | None = None

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": "completed",
            "dataset_id": self.image.dataset_id,
            "split": self.image.split,
            "fold": self.image.fold,
            "image_id": self.image.image_id,
            "original_image_reference": f"{self.image.dataset_id}/{self.image.image_id}",
            "original_size": list(self.image.original_size),
            "fov_valid_fraction": float(np.mean(self.fov_mask > 0)),
            "hair_coverage_fraction": float(np.mean(self.hair_mask > 0)),
            "detections": [detection.manifest() for detection in self.detections],
            "selected_bbox_original": self.selected_bbox_original.as_list() if self.selected_bbox_original else None,
            "expanded_bbox_original": self.expanded_bbox_original.as_list(),
            "coordinate_transform": self.transform.manifest(),
            "detector_failed": self.detector_failed,
            "detector_status": self.detector_status,
            "fallback_used": self.fallback_used,
            "timings_ms": self.timings_ms,
            "warnings": self.warnings,
            "configuration_hash": self.configuration_hash,
            "cache_key": self.cache_key,
            "stages": self.stages,
            "stage_details": self.stage_details,
            "cache_hit": self.cache_hit,
        }


def _odd_scaled(minimum_dimension: int, fraction: float, minimum: int = 3) -> int:
    value = max(minimum, int(round(minimum_dimension * fraction)))
    return value if value % 2 else value + 1


def detect_fov(rgb: np.ndarray, config: dict[str, Any]) -> FOVResult:
    height, width = rgb.shape[:2]
    minimum_dimension = min(height, width)
    kernel = _odd_scaled(minimum_dimension, float(config["median_kernel_fraction"]))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    smooth = cv2.medianBlur(gray, kernel)
    band = max(1, int(round(minimum_dimension * float(config["peripheral_band_fraction"]))))
    peripheral = np.zeros((height, width), dtype=bool)
    peripheral[:band] = True
    peripheral[-band:] = True
    peripheral[:, :band] = True
    peripheral[:, -band:] = True
    inner = ~peripheral
    if not np.any(inner):
        inner[:] = True
    inner_median = float(np.median(smooth[inner]))
    peripheral_low = float(np.percentile(smooth[peripheral], 25))
    contrast = inner_median - peripheral_low
    minimum_contrast = float(config["minimum_relative_contrast"])
    parameters = {**config, "effective_median_kernel": kernel, "effective_band_pixels": band, "observed_contrast": contrast}
    full = np.ones((height, width), dtype=np.uint8)
    if contrast < minimum_contrast:
        return FOVResult(full, True, 1.0, "fallback", ["No hay evidencia periférica suficiente; se usa la imagen completa."], parameters)

    threshold = min(
        inner_median - minimum_contrast,
        max(float(config["absolute_dark_safety"]), float(np.percentile(smooth[peripheral], 50)) + 8.0),
    )
    dark = (smooth <= threshold).astype(np.uint8)
    count, labels = cv2.connectedComponents(dark, connectivity=8)
    border_labels = set(np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))))
    border_labels.discard(0)
    external = np.isin(labels, list(border_labels)) if count > 1 else np.zeros_like(dark, dtype=bool)
    external_fraction = float(np.mean(external))
    if external_fraction < float(config["minimum_external_area_fraction"]):
        return FOVResult(full, True, 1.0, "fallback", ["El candidato exterior es demasiado pequeño; se usa la imagen completa."], parameters)

    valid = (~external).astype(np.uint8)
    closing = _odd_scaled(minimum_dimension, float(config["closing_kernel_fraction"]))
    valid = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, np.ones((closing, closing), np.uint8))
    component_count, component_labels, stats, _ = cv2.connectedComponentsWithStats(valid, connectivity=8)
    center_label = int(component_labels[height // 2, width // 2])
    if center_label == 0 and component_count > 1:
        center_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    valid = (component_labels == center_label).astype(np.uint8)
    valid_fraction = float(np.mean(valid))
    warnings = []
    if valid_fraction < float(config["minimum_valid_area_fraction"]):
        warnings.append("FOV anormalmente pequeño; la política configurada usa imagen completa.")
        return FOVResult(full, True, 1.0, "fallback", warnings, parameters)
    if valid_fraction < 0.5 or valid_fraction > 0.995:
        warnings.append(f"Proporción FOV inusual: {valid_fraction:.4f}")
    return FOVResult(valid, False, valid_fraction, "detected", warnings, parameters)


def _line_kernel(length: int, thickness: int, angle: float) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.uint8)
    center = (length - 1) / 2
    radians = np.deg2rad(angle)
    dx, dy = np.cos(radians) * center, np.sin(radians) * center
    cv2.line(
        kernel,
        (int(round(center - dx)), int(round(center - dy))),
        (int(round(center + dx)), int(round(center + dy))),
        1,
        thickness=max(1, thickness),
    )
    return kernel


def _component_points(labels: np.ndarray, stats: np.ndarray, label: int) -> np.ndarray | None:
    """Return one connected component's global points without scanning its full image."""
    x = int(stats[label, cv2.CC_STAT_LEFT])
    y = int(stats[label, cv2.CC_STAT_TOP])
    width = int(stats[label, cv2.CC_STAT_WIDTH])
    height = int(stats[label, cv2.CC_STAT_HEIGHT])
    if int(stats[label, cv2.CC_STAT_AREA]) < 3:
        return None
    local_y, local_x = np.where(labels[y:y + height, x:x + width] == label)
    return np.column_stack((local_x + x, local_y + y)).astype(np.float32)


def detect_and_inpaint_hair(rgb: np.ndarray, fov_mask: np.ndarray, config: dict[str, Any]) -> HairResult:
    original = rgb.copy()
    height, width = rgb.shape[:2]
    minimum_dimension = min(height, width)
    length = _odd_scaled(minimum_dimension, float(config["line_length_fraction"]), minimum=5)
    thickness = max(1, int(round(minimum_dimension * float(config["line_thickness_fraction"]))))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    responses = [
        cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, _line_kernel(length, thickness, float(angle)))
        for angle in config["orientations_degrees"]
    ]
    response = np.maximum.reduce(responses)
    valid_values = response[fov_mask > 0]
    threshold = max(5.0, float(np.percentile(valid_values, float(config["response_percentile"])))) if valid_values.size else 255.0
    hair = np.zeros((height, width), dtype=np.uint8)
    maximum_thickness = max(2.0, minimum_dimension * float(config["maximum_component_thickness_fraction"]))
    minimum_length = max(3.0, length * 0.35)
    # Filter each orientation before merging so crossing hairs do not become one
    # square component and fail the elongation test.
    for oriented_response in responses:
        candidate = ((oriented_response >= threshold) & (fov_mask > 0)).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
        for label in range(1, count):
            points = _component_points(labels, stats, label)
            if points is None:
                continue
            (_, _), (side_a, side_b), _ = cv2.minAreaRect(points)
            long_side, short_side = max(side_a, side_b), max(1.0, min(side_a, side_b))
            if long_side >= minimum_length and long_side / short_side >= float(config["minimum_elongation"]) and short_side <= maximum_thickness:
                hair[labels == label] = 1
    dilation = _odd_scaled(minimum_dimension, float(config["dilation_fraction"]))
    if np.any(hair):
        hair = cv2.dilate(hair, np.ones((dilation, dilation), np.uint8))
    hair[fov_mask == 0] = 0
    coverage = float(np.mean(hair[fov_mask > 0])) if np.any(fov_mask) else 0.0
    parameters = {**config, "effective_line_length": length, "effective_line_thickness": thickness, "effective_threshold": threshold, "effective_dilation_kernel": dilation}
    if coverage > float(config["maximum_coverage_fraction"]):
        return HairResult(
            np.zeros_like(hair),
            original,
            coverage,
            True,
            [f"Cobertura de vello anormal ({coverage:.4f}); no se aplicó inpainting."],
            parameters,
        )
    if not np.any(hair):
        return HairResult(hair, original, 0.0, False, [], parameters)
    method_name = str(config["inpaint_method"]).lower()
    method = cv2.INPAINT_TELEA if method_name == "telea" else cv2.INPAINT_NS
    radius = max(1.0, minimum_dimension * float(config["inpaint_radius_fraction"]))
    repaired_bgr = cv2.inpaint(cv2.cvtColor(original, cv2.COLOR_RGB2BGR), hair * 255, radius, method)
    repaired = cv2.cvtColor(repaired_bgr, cv2.COLOR_BGR2RGB)
    return HairResult(hair, repaired, coverage, False, [], {**parameters, "effective_inpaint_radius": radius})


def bbox_from_mask(mask: np.ndarray) -> BBox:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return BBox(0, 0, mask.shape[1], mask.shape[0])
    return BBox(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def expand_bbox(bbox: BBox, margin_fraction: float, width: int, height: int) -> BBox:
    margin_x = bbox.width * margin_fraction
    margin_y = bbox.height * margin_fraction
    return BBox(
        max(0, int(np.floor(bbox.x0 - margin_x))),
        max(0, int(np.floor(bbox.y0 - margin_y))),
        min(width, int(np.ceil(bbox.x1 + margin_x))),
        min(height, int(np.ceil(bbox.y1 + margin_y))),
    )


def _image_hash(rgb: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(rgb.shape).encode())
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def atomic_write_png(path: Path, array: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", array)
    if not success:
        raise RuntimeError(f"could not encode {path.name}")
    atomic_write_bytes(path, encoded.tobytes())


def _load_cached(image: ImageInput, directory: Path) -> P0Result | None:
    manifest_path = directory / "preprocessing_manifest.json"
    required = ["fov_mask.png", "hair_mask.png", "segmentation_input.png", "roi_input.png", "roi_fov_mask.png", "yolo_overlay.png", "yolo_bbox.json"]
    if not manifest_path.is_file() or any(not (directory / name).is_file() for name in required):
        return None
    try:
        manifest = load_json(manifest_path)
        if manifest.get("status") != "completed" or manifest.get("cache_key") != directory.name or manifest.get("image_id") != image.image_id:
            return None
        arrays = {name: cv2.imread(str(directory / name), cv2.IMREAD_UNCHANGED) for name in required if name.endswith(".png")}
        if any(value is None for value in arrays.values()):
            return None
        if load_json(directory / "yolo_bbox.json").get("image_id") != image.image_id:
            return None
        selected = manifest.get("selected_bbox_original")
        expanded = BBox.from_list(manifest["expanded_bbox_original"])
        detections = [Detection(BBox.from_list(item["bbox_xyxy_original"]), float(item["confidence"]), item["class_name"]) for item in manifest["detections"]]
        return P0Result(
            image=image,
            fov_mask=(arrays["fov_mask.png"] > 0).astype(np.uint8),
            hair_mask=(arrays["hair_mask.png"] > 0).astype(np.uint8),
            segmentation_input=cv2.cvtColor(arrays["segmentation_input.png"], cv2.COLOR_BGR2RGB),
            detections=detections,
            selected_bbox_original=BBox.from_list(selected) if selected else None,
            expanded_bbox_original=expanded,
            roi_input=cv2.cvtColor(arrays["roi_input.png"], cv2.COLOR_BGR2RGB),
            roi_fov_mask=(arrays["roi_fov_mask.png"] > 0).astype(np.uint8),
            transform=CoordinateTransform(image.original_size[0], image.original_size[1], expanded),
            detector_failed=bool(manifest["detector_failed"]),
            detector_status=str(manifest.get("detector_status", "configured_no_detection" if manifest["detector_failed"] else "configured_success")),
            fallback_used=bool(manifest["fallback_used"]),
            timings_ms={key: float(value) for key, value in manifest["timings_ms"].items()},
            warnings=list(manifest["warnings"]),
            configuration_hash=str(manifest["configuration_hash"]),
            cache_key=str(manifest["cache_key"]),
            stages={key: bool(value) for key, value in manifest.get("stages", {"fov": True, "hair": True, "yolo": True}).items()},
            stage_details=dict(manifest.get("stage_details", {})),
            cache_hit=True,
            cache_directory=directory,
        )
    except (KeyError, OSError, TypeError, ValueError):
        return None


def run_p0(
    image: ImageInput,
    config: dict[str, Any],
    detector: Detector | None = None,
    *,
    cache_root: Path | None = None,
    enable_fov: bool = True,
    enable_hair: bool = True,
    enable_yolo: bool = True,
    reuse_cache: bool = True,
) -> P0Result:
    """Run P0 once and optionally reuse only a complete, hash-matching cache."""
    detector = detector or MissingYoloV3Detector()
    p0_config = config["p0"]
    stages = {"fov": enable_fov, "hair": enable_hair, "yolo": enable_yolo}
    configuration_hash = content_hash({"p0": p0_config, "stages": stages})
    cache_key = content_hash({"image_sha256": _image_hash(image.rgb), "configuration_hash": configuration_hash, "detector": detector.identity})
    cache_directory = cache_root / image.dataset_id / image.image_id / cache_key if cache_root else None
    if cache_directory and reuse_cache:
        cached = _load_cached(image, cache_directory)
        if cached:
            return cached

    timings: dict[str, float] = {}
    started = time.perf_counter()
    fov = detect_fov(image.rgb, p0_config["fov"]) if enable_fov else FOVResult(
        np.ones(image.rgb.shape[:2], np.uint8), False, 1.0, "disabled", [], {"disabled": True}
    )
    timings["fov"] = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    hair = detect_and_inpaint_hair(image.rgb, fov.mask, p0_config["hair"]) if enable_hair else HairResult(
        np.zeros(image.rgb.shape[:2], np.uint8), image.rgb.copy(), 0.0, False, [], {"disabled": True}
    )
    timings["hair"] = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    detector_identity = detector.identity
    detector_available = detector_identity.get("status") != "weights_missing"
    detection_error = None
    try:
        detections = detector.detect(hair.segmentation_input) if enable_yolo else []
    except (cv2.error, RuntimeError) as exc:
        detections = []
        detection_error = f"{type(exc).__name__}: {exc}"
    timings["yolo"] = (time.perf_counter() - started) * 1000
    selected = detections[0].bbox if detections else None
    fallback = enable_yolo and selected is None
    detector_status = (
        "disabled" if not enable_yolo else
        "unavailable" if not detector_available else
        "configured_error" if detection_error else
        "configured_no_detection" if selected is None else
        "configured_success"
    )
    base_bbox = selected or (bbox_from_mask(fov.mask) if enable_yolo else BBox(0, 0, image.original_size[0], image.original_size[1]))
    frozen_margin = detector_identity.get("margin_fraction")
    margin = float(frozen_margin if frozen_margin is not None else p0_config["roi"]["margin_fraction"]) if selected else 0.0
    expanded = expand_bbox(base_bbox, margin, image.original_size[0], image.original_size[1])
    transform = CoordinateTransform(image.original_size[0], image.original_size[1], expanded)
    roi = hair.segmentation_input[expanded.y0:expanded.y1, expanded.x0:expanded.x1].copy()
    roi_fov = fov.mask[expanded.y0:expanded.y1, expanded.x0:expanded.x1].copy()
    warnings = [*fov.warnings, *hair.warnings]
    if fallback:
        warnings.append("YOLOv3 no produjo detección; todos los backends deben usar el ROI completo del FOV.")
    if detection_error:
        warnings.append(f"YOLOv3 falló durante inferencia: {detection_error}")
    result = P0Result(
        image=image,
        fov_mask=fov.mask,
        hair_mask=hair.mask,
        segmentation_input=hair.segmentation_input,
        detections=detections,
        selected_bbox_original=selected,
        expanded_bbox_original=expanded,
        roi_input=roi,
        roi_fov_mask=roi_fov,
        transform=transform,
        detector_failed=fallback,
        detector_status=detector_status,
        fallback_used=fallback,
        timings_ms=timings,
        warnings=warnings,
        configuration_hash=configuration_hash,
        cache_key=cache_key,
        stages=stages,
        stage_details={
            "fov": {
                "confidence": fov.confidence,
                "fallback_full_image": fov.fallback_full_image,
                "valid_fraction": fov.valid_fraction,
                "parameters": fov.parameters,
            },
            "hair": {
                "coverage_fraction": hair.coverage_fraction,
                "fallback_used": hair.fallback_used,
                "parameters": hair.parameters,
            },
            "yolo": {
                "enabled": enable_yolo,
                "detector_identity": detector_identity,
                "detector_status": detector_status,
                "selection_policy": p0_config["yolo"]["selection_policy"],
                "fallback_policy": p0_config["yolo"]["fallback_policy"],
            },
            "roi": {
                "margin_fraction": margin,
                "bbox_convention": "xyxy_half_open",
            },
        },
        cache_hit=False,
        cache_directory=cache_directory,
    )
    if cache_directory:
        atomic_write_png(cache_directory / "fov_mask.png", result.fov_mask * 255)
        atomic_write_png(cache_directory / "hair_mask.png", result.hair_mask * 255)
        atomic_write_png(cache_directory / "segmentation_input.png", cv2.cvtColor(result.segmentation_input, cv2.COLOR_RGB2BGR))
        atomic_write_png(cache_directory / "roi_input.png", cv2.cvtColor(result.roi_input, cv2.COLOR_RGB2BGR))
        atomic_write_png(cache_directory / "roi_fov_mask.png", result.roi_fov_mask * 255)
        yolo_overlay = cv2.cvtColor(image.rgb, cv2.COLOR_RGB2BGR)
        if selected:
            cv2.rectangle(yolo_overlay, (selected.x0, selected.y0), (selected.x1 - 1, selected.y1 - 1), (0, 255, 255), max(1, min(image.rgb.shape[:2]) // 300))
        cv2.rectangle(yolo_overlay, (expanded.x0, expanded.y0), (expanded.x1 - 1, expanded.y1 - 1), (255, 255, 0), max(1, min(image.rgb.shape[:2]) // 300))
        atomic_write_png(cache_directory / "yolo_overlay.png", yolo_overlay)
        atomic_write_json(cache_directory / "yolo_bbox.json", {
            "schema_version": 1,
            "image_id": image.image_id,
            "detections": [item.manifest() for item in detections],
            "selected_detection": detections[0].manifest() if detections else None,
            "bbox_xyxy_original": selected.as_list() if selected else None,
            "confidence": detections[0].confidence if detections else None,
            "margin_fraction": margin,
            "expanded_bbox_xyxy_original": expanded.as_list(),
            "detector_failed": result.detector_failed,
            "detector_status": detector_status,
            "fallback_used": fallback,
            "timing_ms": timings["yolo"],
            "detector_identity": detector_identity,
        })
        atomic_write_json(cache_directory / "preprocessing_manifest.json", result.manifest())
    return result
