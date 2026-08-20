"""Post-Phase-0.5 scientific contracts for final YOLO, model selection, and MSKCC pilot.

This module deliberately does not alter the historical cross-validation artifacts.
The five CV folds remain evidence for out-of-fold estimation and hyperparameter
selection.  A separate ``final_refit`` YOLO checkpoint is required for B1,
component ablations, and every downstream pilot stage.
"""

from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np

from .benchmark import content_hash, load_json, sha256_file


FINAL_YOLO_SCHEMA_VERSION = 3
FINAL_YOLO_ROLE = "final_refit"
COMPONENT_ABLATIONS: dict[str, dict[str, bool]] = {
    "B1_FULL": {"enable_fov": True, "enable_hair": True, "enable_yolo": True, "apply_post": True},
    "B1_NO_HAIR": {"enable_fov": True, "enable_hair": False, "enable_yolo": True, "apply_post": True},
    "B1_NO_YOLO": {"enable_fov": True, "enable_hair": True, "enable_yolo": False, "apply_post": True},
    "B1_NO_FOV": {"enable_fov": False, "enable_hair": True, "enable_yolo": True, "apply_post": True},
    "B1_NO_POST": {"enable_fov": True, "enable_hair": True, "enable_yolo": True, "apply_post": False},
}


def _resolve(path_value: str | Path, root: Path | None = None) -> Path:
    path = Path(path_value)
    if not path.is_absolute() and root is not None:
        path = root / path
    return path.resolve()


def scaled_final_batches(final_count: int, folds_payload: dict[str, Any], cv_max_batches: int) -> int:
    """Preserve approximate training exposure when refitting on all CV training data.

    Each CV model sees only its fold's training partition.  If CV trained for
    ``cv_max_batches`` batches, the final refit scales the number of batches by
    ``N_final / mean(N_fold_train)`` so the number of sample-exposures per image
    stays approximately constant.  No test or held-out development images are
    introduced by this calculation.
    """
    if final_count <= 0 or cv_max_batches <= 0:
        raise ValueError("final_count and cv_max_batches must be positive")
    entries = folds_payload.get("folds") or []
    if len(entries) != 5 or {entry.get("fold") for entry in entries} != set(range(5)):
        raise ValueError("Expected exactly five CV folds numbered 0..4")
    train_sizes = [len(entry.get("train_ids") or []) for entry in entries]
    if any(size <= 0 for size in train_sizes):
        raise ValueError("Every CV fold must contain training ids")
    mean_train = mean(train_sizes)
    return max(1, int(round(cv_max_batches * final_count / mean_train)))


def validate_final_yolo_manifest(
    manifest_or_path: dict[str, Any] | Path,
    *,
    verify_artifacts: bool = True,
) -> dict[str, Any]:
    """Validate a frozen YOLO final-refit manifest and, optionally, its artifacts."""
    payload = deepcopy(load_json(manifest_or_path) if isinstance(manifest_or_path, Path) else manifest_or_path)
    identity = payload.pop("identity_hash", None)
    if payload.get("schema_version") != FINAL_YOLO_SCHEMA_VERSION:
        raise ValueError("Final YOLO manifest has an unsupported schema")
    if payload.get("status") != "frozen" or payload.get("architecture") != "YOLOv3-Darknet53":
        raise ValueError("Final YOLO manifest is not a frozen YOLOv3-Darknet53 artifact")
    if payload.get("model_role") != FINAL_YOLO_ROLE:
        raise ValueError("B1 requires model_role=final_refit; a CV fold cannot be used")
    if payload.get("fold") is not None:
        raise ValueError("A final-refit checkpoint must not masquerade as a CV fold")
    thresholds = payload.get("thresholds") or {}
    required_thresholds = {"confidence_threshold", "nms_threshold", "margin_fraction"}
    if not required_thresholds <= set(thresholds):
        raise ValueError("Final YOLO manifest is missing frozen thresholds")
    if content_hash(payload) != identity:
        raise ValueError("Final YOLO identity hash is invalid")
    payload["identity_hash"] = identity
    if verify_artifacts:
        for name in ("cfg", "weights", "training_state"):
            path = Path(payload.get(f"{name}_path", ""))
            expected = payload.get(f"{name}_sha256")
            if not path.is_file():
                raise FileNotFoundError(f"Missing final YOLO {name}: {path}")
            if sha256_file(path) != expected:
                raise ValueError(f"Final YOLO {name} hash changed: {path}")
    return payload


def build_final_b1_config(
    base_config: dict[str, Any],
    frozen: dict[str, Any],
    *,
    artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    """Create the only supported B1 configuration after Phase 0.5.

    ``detector_from_config`` already supports direct cfg/weights paths.  We use
    that path intentionally so historical schema-2 CV manifests remain untouched,
    while the generated configuration records the final-refit identity explicitly.
    """
    frozen = validate_final_yolo_manifest(frozen, verify_artifacts=True)
    config = deepcopy(base_config)
    yolo = config["p0"]["yolo"]
    yolo.update(
        {
            "frozen_manifest": None,
            "cfg_path": frozen["cfg_path"],
            "weights_path": frozen["weights_path"],
            "confidence_threshold": float(frozen["thresholds"]["confidence_threshold"]),
            "nms_threshold": float(frozen["thresholds"]["nms_threshold"]),
            "model_role": FINAL_YOLO_ROLE,
            "final_frozen_identity_hash": frozen["identity_hash"],
            "final_frozen_manifest_path": frozen.get("manifest_path"),
        }
    )
    config["p0"]["roi"]["margin_fraction"] = float(frozen["thresholds"]["margin_fraction"])
    config["post05_contract"] = {
        "schema_version": 1,
        "phase": "post_0_5",
        "yolo_model_role": FINAL_YOLO_ROLE,
        "yolo_identity_hash": frozen["identity_hash"],
        "cv_fold_checkpoint_allowed_for_b1": False,
    }
    if artifact_root is not None:
        config["artifact_root"] = str(artifact_root)
    return config


def require_final_b1_config(config: dict[str, Any]) -> None:
    yolo = config.get("p0", {}).get("yolo", {})
    contract = config.get("post05_contract") or {}
    if yolo.get("model_role") != FINAL_YOLO_ROLE:
        raise ValueError("Post-0.5 B1/ablation requires the final-refit YOLO configuration")
    if contract.get("yolo_model_role") != FINAL_YOLO_ROLE:
        raise ValueError("Missing post05 final-refit contract")
    if not yolo.get("final_frozen_identity_hash") or yolo.get("final_frozen_identity_hash") != contract.get("yolo_identity_hash"):
        raise ValueError("Final-refit YOLO identity is missing or inconsistent")
    if yolo.get("frozen_manifest"):
        raise ValueError("Post-0.5 B1 must not point at a per-fold frozen_manifest")


def _finite(values: Iterable[Any]) -> np.ndarray:
    return np.asarray([float(value) for value in values if value is not None and math.isfinite(float(value))], dtype=float)


def _rows_for(rows: list[dict[str, Any]], method_id: str, condition: str | None = None) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row.get("method_id") == method_id
        and (condition is None or row.get("condition") == condition)
    ]


def compute_segmenter_score(
    full_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    method_id: str,
) -> dict[str, Any]:
    """Compute the pre-registered 100-point TOP-3 score for one segmenter."""
    base = _rows_for(full_rows, method_id)
    if not base:
        raise ValueError(f"No B1_FULL rows for {method_id}")
    scientific = [row for row in base if not bool(row.get("technical_failure"))]
    if not scientific:
        raise ValueError(f"No scientific B1_FULL rows for {method_id}")

    dice = _finite(row.get("dice") for row in scientific)
    jaccard = _finite(row.get("jaccard") for row in scientific)
    boundary = _finite(row.get("boundary_f1") for row in scientific)
    hd95 = _finite(row.get("hd95_normalized") for row in scientific)
    if not len(dice) or not len(jaccard) or not len(boundary):
        raise ValueError(f"Required region/boundary metrics missing for {method_id}")

    dice_mean = float(np.mean(dice))
    jaccard_mean = float(np.mean(jaccard))
    boundary_mean = float(np.mean(boundary))
    hd95_mean = float(np.mean(hd95)) if len(hd95) else float("inf")
    hd95_quality = 0.0 if not math.isfinite(hd95_mean) else 1.0 / (1.0 + max(0.0, hd95_mean))

    q75, q25 = np.percentile(dice, [75, 25]) if len(dice) > 1 else (dice_mean, dice_mean)
    dice_iqr = float(q75 - q25)
    stability = float(np.clip(1.0 - dice_iqr, 0.0, 1.0))

    technical_rate = sum(bool(row.get("technical_failure")) for row in base) / len(base)
    degenerate_rate = sum(bool(row.get("degenerate_prediction")) for row in base) / len(base)
    reliability = float(np.clip(1.0 - technical_rate - 0.75 * degenerate_rate, 0.0, 1.0))

    condition_means: dict[str, float] = {}
    for condition in COMPONENT_ABLATIONS:
        values = _finite(
            row.get("dice")
            for row in _rows_for(ablation_rows, method_id, condition)
            if not bool(row.get("technical_failure"))
        )
        if len(values):
            condition_means[condition] = float(np.mean(values))
    if set(condition_means) != set(COMPONENT_ABLATIONS):
        missing = sorted(set(COMPONENT_ABLATIONS) - set(condition_means))
        raise ValueError(f"Missing ablation Dice for {method_id}: {missing}")
    condition_values = np.asarray([condition_means[name] for name in COMPONENT_ABLATIONS], dtype=float)
    full_ablation_mean = condition_means["B1_FULL"]
    ablation_delta_from_full = {name: full_ablation_mean - value for name, value in condition_means.items()}
    robustness = float(np.clip(np.mean(condition_values) - np.std(condition_values), 0.0, 1.0))

    components = {
        "region": 17.5 * dice_mean + 17.5 * jaccard_mean,
        "boundary": 12.0 * boundary_mean + 8.0 * hd95_quality,
        "stability": 15.0 * stability,
        "reliability": 15.0 * reliability,
        "p0_robustness": 15.0 * robustness,
    }
    score = float(sum(components.values()))
    return {
        "method_id": method_id,
        "score": score,
        "components": components,
        "raw": {
            "dice_mean": dice_mean,
            "jaccard_mean": jaccard_mean,
            "boundary_f1_mean": boundary_mean,
            "hd95_normalized_mean": None if not math.isfinite(hd95_mean) else hd95_mean,
            "hd95_quality": hd95_quality,
            "dice_iqr": dice_iqr,
            "stability": stability,
            "technical_failure_rate": technical_rate,
            "degenerate_prediction_rate": degenerate_rate,
            "reliability": reliability,
            "ablation_dice_means": condition_means,
            "ablation_delta_dice_from_full": ablation_delta_from_full,
            "p0_robustness": robustness,
        },
    }


def choose_top3(
    full_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    *,
    max_technical_failure_rate: float = 0.01,
    max_degenerate_rate: float = 0.10,
) -> list[dict[str, Any]]:
    methods = sorted({str(row["method_id"]) for row in full_rows})
    scored: list[dict[str, Any]] = []
    for method in methods:
        item = compute_segmenter_score(full_rows, ablation_rows, method)
        technical = item["raw"]["technical_failure_rate"]
        degenerate = item["raw"]["degenerate_prediction_rate"]
        reasons = []
        if technical > max_technical_failure_rate:
            reasons.append(f"technical_failure_rate>{max_technical_failure_rate:.3f}")
        if degenerate > max_degenerate_rate:
            reasons.append(f"degenerate_prediction_rate>{max_degenerate_rate:.3f}")
        item["eligible"] = not reasons
        item["exclusion_reasons"] = reasons
        scored.append(item)
    scored.sort(key=lambda row: (not row["eligible"], -float(row["score"]), row["method_id"]))
    if sum(bool(row["eligible"]) for row in scored) < 3:
        raise ValueError("Fewer than three segmenters pass the pre-registered reliability gate")
    return scored


COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "image_id": ("image_id", "isic_id", "image", "id"),
    "patient_id": ("patient_id", "patient", "subject_id", "subject"),
    "lesion_id": ("lesion_id", "lesion", "lesion_identifier"),
    "mst": ("mst", "monk_skin_tone", "monk", "monk_tone"),
    "fst": ("fst", "fitzpatrick", "fitzpatrick_skin_type", "fitzpatrick_type"),
    "l_star": ("l_star", "lstar", "l*", "lab_l"),
    "a_star": ("a_star", "astar", "a*", "lab_a"),
    "b_star": ("b_star", "bstar", "b*", "lab_b"),
    "ita": ("ita", "individual_typology_angle", "ita_degrees"),
    "acquisition_mode": ("acquisition_mode", "image_type", "modality", "acquisition"),
    "anatomical_site": ("anatomical_site", "site", "body_site", "anatom_site_general"),
}


def _normalized_header(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def infer_column_map(fieldnames: Iterable[str], explicit: dict[str, str] | None = None) -> dict[str, str]:
    normalized = {_normalized_header(name): name for name in fieldnames}
    result: dict[str, str] = {}
    explicit = explicit or {}
    for canonical, source in explicit.items():
        if source not in fieldnames:
            raise ValueError(f"Explicit metadata column not found: {source}")
        result[canonical] = source
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in result:
            continue
        for alias in aliases:
            source = normalized.get(_normalized_header(alias))
            if source is not None:
                result[canonical] = source
                break
    return result


def parse_optional_float(value: Any) -> float | None:
    if value in (None, "", "NA", "N/A", "nan", "NaN", "null", "None"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_optional_int(value: Any) -> int | None:
    number = parse_optional_float(value)
    if number is None or not float(number).is_integer():
        return None
    return int(number)


def read_metadata_table(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("items", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("JSON metadata must be a list of objects or {'items': [...]} payload")
        fields = sorted({key for row in rows for key in row})
        return rows, fields
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError("Metadata CSV has no header")
        return list(reader), list(reader.fieldnames)


def stable_tiebreak(seed: int, *parts: str) -> str:
    return hashlib.sha256((str(seed) + ":" + "|".join(parts)).encode()).hexdigest()
