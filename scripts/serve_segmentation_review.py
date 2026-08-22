"""Serve the local segmentation comparison and mask-review interface."""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import mimetypes
import os
from pathlib import Path
import resource
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from PIL import Image

from _project_paths import REPO_ROOT, require_within

sys.path.insert(0, str(REPO_ROOT / "src"))
from thesis_fitzpatrick.masks import (  # noqa: E402
    build_clean_skin_mask,
    measure_skin_colour,
    normalize_binary_mask,
    save_binary_mask,
)
from thesis_fitzpatrick.benchmark import (  # noqa: E402
    evaluation_registry,
    is_fatal_backend_failure,
    load_benchmark_methods,
    load_json,
    content_hash,
    sha256_file,
    validate_dataset_registry,
)
from thesis_fitzpatrick.annotations import project_status, save_mask_version  # noqa: E402


WEB_DIR = REPO_ROOT / "web"
MODEL_CONFIG = REPO_ROOT / "configs" / "segmentation_models.json"
PILOT_INPUT_DIR = REPO_ROOT / "data" / "processed" / "isic_segmentation_pilot_resized_inference"
FITZPATRICK_INPUT_DIR = REPO_ROOT / "data" / "raw" / "isic_fitzpatrick_images"
INPUT_ROOTS = {
    "pilot": PILOT_INPUT_DIR,
    "fitzpatrick": FITZPATRICK_INPUT_DIR,
}
METADATA_CSV = REPO_ROOT / "data" / "raw" / "isic_fitzpatrick_metadata_full.csv"
RESULTS_DIR = REPO_ROOT / "results" / "segmentation_benchmark"
BENCHMARK_CONFIG = REPO_ROOT / "configs" / "benchmark" / "default.json"
DATASET_CONFIG = REPO_ROOT / "configs" / "benchmark" / "datasets.json"
BENCHMARK_ARTIFACT_ROOT = REPO_ROOT / "results" / "benchmark_v1"
ANNOTATION_PROJECT = REPO_ROOT / "data" / "interim" / "fitzpatrick_annotations_v1"
POST05_REVIEW_DIR = BENCHMARK_ARTIFACT_ROOT / "review"
POST05_OBSERVATIONS = POST05_REVIEW_DIR / "post05_observations.jsonl"
POST05_FREEZE = BENCHMARK_ARTIFACT_ROOT / "post05_scientific_freeze.json"
POST05_SELECTION = BENCHMARK_ARTIFACT_ROOT / "post05_selection" / "segmenter_selection.json"
POST05_YOLO_FREEZE = BENCHMARK_ARTIFACT_ROOT / "yolo" / "final" / "frozen.json"
POST05_CONDITIONS = ("B1_FULL", "B1_NO_HAIR", "B1_NO_YOLO", "B1_NO_FOV", "B1_NO_POST")
POST05_TOP3 = ("S01", "S10", "S14")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
FITZPATRICK_TYPES = ("I", "II", "III", "IV", "V", "VI")


def load_models() -> list[dict]:
    payload = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
    return sorted(payload["models"], key=lambda model: model["priority"])


def benchmark_state() -> dict:
    methods = load_benchmark_methods(MODEL_CONFIG)
    datasets = validate_dataset_registry(load_json(DATASET_CONFIG))
    config = load_json(BENCHMARK_CONFIG)
    yolo = config["p0"]["yolo"]
    yolo_paths = [yolo.get("cfg_path"), yolo.get("weights_path")]
    frozen_yolo = yolo.get("frozen_manifest")
    yolo_error = None
    yolo_ready = all(
        value and (REPO_ROOT / value).is_file()
        for value in yolo_paths
    ) and yolo.get("confidence_threshold") is not None and yolo.get("nms_threshold") is not None
    if frozen_yolo and (REPO_ROOT / frozen_yolo).is_file():
        try:
            frozen_detector = load_json(REPO_ROOT / frozen_yolo)
            yolo_ready = frozen_detector.get("status") == "frozen" and all(
                (Path(value) if Path(value).is_absolute() else REPO_ROOT / value).is_file()
                for value in (frozen_detector.get("cfg_path", ""), frozen_detector.get("weights_path", ""))
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            yolo_ready, yolo_error = False, str(exc)
    b2_path = BENCHMARK_ARTIFACT_ROOT / "b2_frozen.json"
    missing_b2 = [method["method_id"] for method in methods if method["kind"] == "neural"]
    if b2_path.is_file():
        try:
            frozen_b2 = load_json(b2_path)
            identities = frozen_b2.get("b2_checkpoints", {}) if frozen_b2.get("frozen") is True else {}
            missing_b2 = []
            for method in (item for item in methods if item["kind"] == "neural"):
                members = (identities.get(method["method_id"]) or {}).get("members", [])
                if len(members) not in {1, 5} or any(not (Path(member["path"]) if Path(member["path"]).is_absolute() else REPO_ROOT / member["path"]).is_file() for member in members):
                    missing_b2.append(method["method_id"])
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    evaluations = evaluation_registry(methods)
    b2_evaluation = next(item for item in evaluations if item["id"] == "B2")
    if not missing_b2:
        b2_evaluation.update({"available": True, "reason": None, "missing_methods": []})
    else:
        b2_evaluation["missing_methods"] = missing_b2
    backend_resources = {}
    for method in methods:
        path = BENCHMARK_ARTIFACT_ROOT / "resources" / f"{method['method_id']}.json"
        if path.is_file(): backend_resources[method["method_id"]] = load_json(path)
    return {
        "schema_version": 1,
        "methods": methods,
        "evaluations": evaluations,
        "datasets": datasets,
        "implementation": {
            "contracts_and_registries": "ready",
            "p0": "ready_with_yolo_fallback",
            "s16_classic": "ready",
            "s16_robust": "ready",
            "metrics": "ready",
            "native_review_ui": "ready",
            "ablation_s01_s15": "ready",
            "b2_training": "ready_pending_training_runs",
            "sealed_execution": "ready_and_locked",
        },
        "resources": {
            "yolov3": {
                "available": yolo_ready,
                "architecture": yolo["architecture"],
                "message": "Disponible" if yolo_ready else yolo_error or "Faltan cfg/pesos YOLOv3 locales; P0 usa fallback FOV registrado.",
            },
            "b2_checkpoints": {"available": not missing_b2, "missing_count": len(missing_b2), "missing_methods": missing_b2, "registry": str(b2_path.relative_to(REPO_ROOT))},
            "isic2018_task1_manifest": {
                "available": any((REPO_ROOT / "data" / "raw" / "isic2018_task1" / "manifests").glob("isic2018_task1_*.json")),
            },
            "backends": backend_resources,
        },
    }


def benchmark_runs() -> list[dict]:
    runs = []
    for path in sorted((BENCHMARK_ARTIFACT_ROOT / "runs").glob("*/run_manifest.json"), reverse=True):
        try:
            manifest = load_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        runs.append({key: manifest.get(key) for key in ("run_id", "status", "evaluation", "conditions", "dataset", "split", "methods", "images", "created_utc", "completed_utc")})
    return runs


def benchmark_run(run_id: str) -> dict:
    if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for char in run_id):
        raise ValueError("run_id inválido")
    root = require_within(BENCHMARK_ARTIFACT_ROOT / "runs" / run_id, (BENCHMARK_ARTIFACT_ROOT / "runs",), "Run")
    manifest = load_json(root / "run_manifest.json")
    report = load_json(root / "report.json") if (root / "report.json").is_file() else None
    local_image_urls = {item["id"]: item["url"] for item in list_images()}
    results = []
    for path in sorted((root / "predictions").rglob("result.json")):
        payload = load_json(path)
        failure_code = payload.get("failure_code") or (payload.get("backend") or {}).get("failure_code")
        failure_is_fatal = is_fatal_backend_failure(failure_code)
        flags = (payload.get("metrics") or {}).get("flags") or {}
        degenerate = (not failure_is_fatal) and bool(
            failure_code
            or flags.get("prediction_empty")
            or flags.get("prediction_nearly_complete")
        )
        payload["failure_is_fatal"] = failure_is_fatal
        payload["outcome"] = (
            "technical_failure" if failure_is_fatal
            else "degenerate_prediction" if degenerate
            else "ok"
        )
        relative_root = path.parent.relative_to(root).as_posix()
        artifacts = {}
        if payload.get("original_preview") and (root / payload["original_preview"]).is_file():
            artifacts["original_image.jpg"] = f"/files/benchmark/{run_id}/{payload['original_preview']}"
        elif payload.get("image_id") in local_image_urls:
            artifacts["original_image.jpg"] = local_image_urls[payload["image_id"]]
        elif (root / "inputs" / f"{payload.get('image_id')}.jpg").is_file():
            artifacts["original_image.jpg"] = f"/files/benchmark/{run_id}/inputs/{payload['image_id']}.jpg"
        if payload.get("ground_truth_preview") and (root / payload["ground_truth_preview"]).is_file(): artifacts["ground_truth.png"] = f"/files/benchmark/{run_id}/{payload['ground_truth_preview']}"
        elif (root / "inputs" / f"{payload.get('image_id')}.ground_truth.png").is_file(): artifacts["ground_truth.png"] = f"/files/benchmark/{run_id}/inputs/{payload['image_id']}.ground_truth.png"
        if payload.get("overlay_preview") and (path.parent / payload["overlay_preview"]).is_file(): artifacts["prediction_overlay.jpg"] = f"/files/benchmark/{run_id}/{relative_root}/{payload['overlay_preview']}"
        for name in ("native_mask.png", "pre_postprocess_mask.png", "final_mask.png", "clean_skin_mask.png", "raw_probability.npy"):
            if (path.parent / name).is_file(): artifacts[name] = f"/files/benchmark/{run_id}/{relative_root}/{name}"
        cache_key = payload.get("p0_cache_key")
        if cache_key:
            p0_root = BENCHMARK_ARTIFACT_ROOT / "preprocessing" / str(payload.get("dataset_id") or manifest.get("dataset")) / str(payload["image_id"]) / str(cache_key)
            for name in ("fov_mask.png", "hair_mask.png", "segmentation_input.png", "yolo_overlay.png", "roi_input.png", "yolo_bbox.json"):
                if (p0_root / name).is_file(): artifacts[name] = f"/files/artifact/preprocessing/{payload.get('dataset_id') or manifest.get('dataset')}/{payload['image_id']}/{cache_key}/{name}"
        payload["artifacts"] = artifacts
        payload["review_identity"] = {
            "run_id": run_id,
            "method_id": payload.get("method_id"),
            "image_id": payload.get("image_id"),
            "condition": payload.get("condition") or payload.get("evaluation"),
            "hashes": {
                key: value for key, value in {
                    "configuration_hash": manifest.get("configuration_hash"),
                    "dataset_manifest_hash": manifest.get("dataset_manifest_hash"),
                    "image_sha256": (payload.get("metadata") or {}).get("image_sha256"),
                    "mask_sha256": (payload.get("metadata") or {}).get("mask_sha256"),
                    "yolo_final_identity_hash": manifest.get("yolo_final_identity_hash"),
                }.items() if value
            },
        }
        results.append(payload)
    return {"manifest": manifest, "report": report, "results": results}


def _freeze_path(path: Path, label: str) -> Path:
    if path.is_file():
        return path
    raise ValueError(f"No se encontró {label}: {path}")


def _run_path(value: str, label: str) -> Path:
    candidate = Path(value)
    if candidate.is_file():
        return candidate
    candidate = BENCHMARK_ARTIFACT_ROOT / "runs" / candidate.name
    if not (candidate / "run_manifest.json").is_file():
        raise ValueError(f"No se encontró el run congelado de {label}: {value}")
    return candidate


def _without_identity(payload: dict) -> dict:
    copy = dict(payload)
    copy.pop("identity_hash", None)
    return copy


def post05_contract() -> dict:
    freeze_path = _freeze_path(POST05_FREEZE, "scientific freeze")
    selection_path = _freeze_path(POST05_SELECTION, "selección congelada")
    yolo_path = _freeze_path(POST05_YOLO_FREEZE, "YOLO final congelado")
    freeze = load_json(freeze_path)
    selection = load_json(selection_path)
    yolo = load_json(yolo_path)
    if freeze.get("status") != "frozen_before_mskcc" or freeze.get("schema_version") != 1:
        raise ValueError("El scientific freeze post-0.5 no está congelado o tiene un schema incompatible")
    if freeze.get("identity_hash") != content_hash(_without_identity(freeze)):
        raise ValueError("identity_hash del scientific freeze no coincide")
    if selection.get("identity_hash") != content_hash(_without_identity(selection)):
        raise ValueError("identity_hash de la selección congelada no coincide")
    if yolo.get("identity_hash") != content_hash(_without_identity(yolo)):
        raise ValueError("identity_hash del YOLO final no coincide")
    if sha256_file(selection_path) != freeze.get("selection_manifest_sha256"):
        raise ValueError("El hash de la selección no coincide con el scientific freeze")
    if sha256_file(yolo_path) != freeze.get("final_yolo_manifest_sha256"):
        raise ValueError("El hash del manifest YOLO no coincide con el scientific freeze")
    if tuple(freeze.get("top3", ())) != POST05_TOP3 or tuple(selection.get("top3", ())) != POST05_TOP3:
        raise ValueError("TOP-3 congelado incompatible: se esperaban S01, S10 y S14")
    if selection.get("identity_hash") != freeze.get("selection_identity_hash"):
        raise ValueError("La identidad de selección no coincide con el scientific freeze")
    if yolo.get("identity_hash") != freeze.get("final_yolo_identity_hash") or yolo.get("identity_hash") != selection.get("yolo_final_identity_hash"):
        raise ValueError("La identidad del YOLO final no coincide con los freezes")
    if tuple(freeze.get("ablation_conditions", ())) != POST05_CONDITIONS:
        raise ValueError("Los cinco tipos de ablación no coinciden con el scientific freeze")

    run_specs = {}
    expected_ablation_hashes = selection.get("ablation_run_manifest_sha256") or []
    for index, value in enumerate(selection.get("ablation_runs") or []):
        run_path = _run_path(value, f"ablación {index + 1}")
        manifest_path = run_path / "run_manifest.json"
        if index >= len(expected_ablation_hashes) or sha256_file(manifest_path) != expected_ablation_hashes[index]:
            raise ValueError(f"El hash del manifest de ablación no coincide: {run_path.name}")
        manifest = load_json(manifest_path)
        condition = (manifest.get("conditions") or [None])[0]
        if manifest.get("status") != "completed" or manifest.get("evaluation") != "post05_component_ablation" or condition not in POST05_CONDITIONS:
            raise ValueError(f"Run post-0.5 inválido o incompleto: {run_path.name}")
        if manifest.get("yolo_final_identity_hash") != yolo.get("identity_hash"):
            raise ValueError(f"YOLO final inconsistente en {run_path.name}")
        if condition in run_specs:
            raise ValueError(f"Hay más de un run congelado para {condition}")
        run_specs[condition] = {"run_id": run_path.name, "path": run_path, "manifest": manifest, "manifest_sha256": expected_ablation_hashes[index]}
    if set(run_specs) != set(POST05_CONDITIONS):
        raise ValueError(f"Faltan runs post-0.5 congelados: {sorted(set(POST05_CONDITIONS) - set(run_specs))}")
    return {
        "schema_version": 1,
        "freeze_identity_hash": freeze["identity_hash"],
        "selection_identity_hash": selection["identity_hash"],
        "yolo_final_identity_hash": yolo["identity_hash"],
        "top3": list(POST05_TOP3),
        "top3_resources": freeze.get("top3_resources", {}),
        "conditions": list(POST05_CONDITIONS),
        "runs": {condition: {key: value for key, value in spec.items() if key != "path" and key != "manifest"} | {"run_id": spec["run_id"]} for condition, spec in run_specs.items()},
        "run_paths": run_specs,
        "freeze": freeze,
        "selection": selection,
        "yolo": yolo,
    }


def post05_observations(contract: dict | None = None, cases: dict[tuple[str, str], dict] | None = None) -> dict:
    contract = contract or post05_contract()
    cases = cases or _post05_case_index(contract)
    history = []
    if POST05_OBSERVATIONS.is_file():
        for line in POST05_OBSERVATIONS.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if value.get("schema_version") in {1, 2}:
                history.append(value)
    current_integrity = {
        "freeze_identity_hash": contract["freeze_identity_hash"],
        "selection_identity_hash": contract["selection_identity_hash"],
        "yolo_final_identity_hash": contract["yolo_final_identity_hash"],
    }
    valid_keys = set(cases)
    current_history = []
    for value in history:
        if value.get("schema_version") != 2:
            continue
        if (value.get("method_id"), value.get("image_id")) not in valid_keys:
            continue
        integrity = value.get("integrity") or {}
        if any(integrity.get(key) != expected for key, expected in current_integrity.items()):
            continue
        current_history.append(value)
    latest = {}
    for value in current_history:
        key = (value.get("method_id"), value.get("image_id"))
        if key[0] and key[1]:
            latest["::".join(key)] = value
    return {"history": history, "current_history": current_history, "latest": latest, "reviewed": len(latest)}


def _post05_case_index(contract: dict) -> dict[tuple[str, str], dict]:
    cases: dict[tuple[str, str], dict] = {}
    for condition in POST05_CONDITIONS:
        run_id = contract["run_paths"][condition]["run_id"]
        payload = benchmark_run(run_id)
        for item in payload["results"]:
            if item.get("method_id") not in POST05_TOP3:
                continue
            key = (item["method_id"], item["image_id"])
            entry = cases.setdefault(key, {"method_id": item["method_id"], "image_id": item["image_id"], "conditions": {}})
            entry["conditions"][condition] = {
                "run_id": run_id,
                "metrics": item.get("metrics"),
                "artifacts": item.get("artifacts"),
                "review_identity": item.get("review_identity"),
                "hashes": {
                    "manifest_sha256": contract["run_paths"][condition]["manifest_sha256"],
                    "configuration_hash": contract["run_paths"][condition]["manifest"].get("configuration_hash"),
                    "dataset_manifest_hash": contract["run_paths"][condition]["manifest"].get("dataset_manifest_hash"),
                    "image_sha256": (item.get("metadata") or {}).get("image_sha256"),
                    "mask_sha256": (item.get("metadata") or {}).get("mask_sha256"),
                    "yolo_final_identity_hash": contract["yolo_final_identity_hash"],
                    "freeze_identity_hash": contract["freeze_identity_hash"],
                    "selection_identity_hash": contract["selection_identity_hash"],
                },
            }
    expected = len(POST05_TOP3) * 64
    if len(cases) != expected or any(set(value["conditions"]) != set(POST05_CONDITIONS) for value in cases.values()):
        raise ValueError(f"Cobertura post-0.5 incompleta: {len(cases)} casos; se esperaban {expected} con cinco condiciones")
    return cases


def post05_state() -> dict:
    contract = post05_contract()
    cases = _post05_case_index(contract)
    selection_rows = {row["method_id"]: row for row in contract["selection"].get("ranking", []) if row.get("method_id") in POST05_TOP3}
    return {
        "contract": {key: value for key, value in contract.items() if key not in {"run_paths", "freeze", "selection", "yolo"}},
        "summary": [selection_rows[method] for method in POST05_TOP3],
        "cases": list(cases.values()),
        "observations": post05_observations(contract, cases),
    }


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    binary = mask > 0
    inner = binary.copy()
    inner[1:] &= binary[:-1]
    inner[:-1] &= binary[1:]
    inner[:, 1:] &= binary[:, :-1]
    inner[:, :-1] &= binary[:, 1:]
    return binary & ~inner


def post05_overlay(query: dict) -> bytes:
    contract = post05_contract()
    condition = query.get("condition", "B1_FULL")
    if condition not in POST05_CONDITIONS:
        raise ValueError("condición post-0.5 inválida")
    if query.get("method_id") not in POST05_TOP3:
        raise ValueError("method_id no pertenece al TOP-3 congelado")
    run_id = contract["run_paths"][condition]["run_id"]
    root = contract["run_paths"][condition]["path"]
    image_id = query.get("image_id", "")
    result_path = next(root.glob(f"predictions/{condition}/{query['method_id']}/{image_id}/result.json"), None)
    if result_path is None:
        raise ValueError("caso post-0.5 no encontrado")
    result = load_json(result_path)
    original = Image.open(root / result.get("original_preview", f"inputs/{image_id}.jpg")).convert("RGB")
    gt = Image.open(root / result.get("ground_truth_preview", f"inputs/{image_id}.ground_truth.png")).convert("L").resize(original.size, Image.Resampling.NEAREST)
    pred = Image.open(result_path.parent / "final_mask.png").convert("L").resize(original.size, Image.Resampling.NEAREST)
    canvas = np.asarray(original).copy()
    for mask, color in ((_mask_boundary(np.asarray(gt)), (0, 255, 0)), (_mask_boundary(np.asarray(pred)), (255, 64, 32))):
        canvas[mask] = (0.35 * canvas[mask] + 0.65 * np.asarray(color)).astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(canvas).save(stream, format="PNG")
    return stream.getvalue()


def save_post05_observation(request: dict) -> dict:
    required = ("actor", "method_id", "image_id", "conditions", "assessment")
    if any(not request.get(key) for key in required):
        raise ValueError("actor, method_id, image_id, conditions y assessment son obligatorios")
    contract = post05_contract()
    cases = _post05_case_index(contract)
    key = (str(request["method_id"]), str(request["image_id"]))
    if key not in cases:
        raise ValueError("method_id + image_id no pertenecen a los 192 casos congelados")
    conditions = request["conditions"]
    if set(conditions) != set(POST05_CONDITIONS):
        raise ValueError("La observación debe conservar las cinco condiciones post-0.5")
    for condition in POST05_CONDITIONS:
        if conditions[condition].get("run_id") != contract["run_paths"][condition]["run_id"]:
            raise ValueError(f"run_id inconsistente para {condition}")
    assessment = request["assessment"]
    allowed = {
        "roi_yolo_issue": {"yes", "no", "uncertain"},
        "visual_preference": {"FULL", "NO_FOV", "NO_POST", "tie_unclear"},
        "general_status": {"reviewed_ok", "needs_attention"},
    }
    for key_name, values in allowed.items():
        if assessment.get(key_name) not in values:
            raise ValueError(f"assessment.{key_name} inválido")
    for key_name in ("fov_clipping_severity", "postprocessing_damage_severity", "no_post_artifact_severity"):
        if assessment.get(key_name) not in {0, 1, 2, 3}:
            raise ValueError(f"assessment.{key_name} debe ser 0, 1, 2 o 3")
    if not str(assessment.get("note", "")).strip():
        raise ValueError("assessment.note es obligatorio")
    canonical_conditions = {
        condition: {"run_id": contract["run_paths"][condition]["run_id"], "hashes": cases[key]["conditions"][condition]["hashes"]}
        for condition in POST05_CONDITIONS
    }
    record = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "actor": str(request["actor"]).strip(),
        "method_id": key[0],
        "image_id": key[1],
        "conditions": canonical_conditions,
        "assessment": {**assessment, "note": str(assessment["note"]).strip()},
        "integrity": {"freeze_identity_hash": contract["freeze_identity_hash"], "selection_identity_hash": contract["selection_identity_hash"], "yolo_final_identity_hash": contract["yolo_final_identity_hash"]},
    }
    POST05_REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    with POST05_OBSERVATIONS.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return record


def load_image_metadata() -> dict[str, dict]:
    if not METADATA_CSV.is_file():
        return {}
    metadata = {}
    with METADATA_CSV.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            image_id = str(row.get("isic_id", "")).strip()
            if not image_id or image_id in metadata:
                continue
            metadata[image_id] = {
                "fitzpatrick_skin_type": str(row.get("fitzpatrick_skin_type", "")).strip(),
                "diagnosis_1": str(row.get("diagnosis_1", "")).strip(),
                "image_type": str(row.get("image_type", "")).strip(),
                "copyright_license": str(row.get("copyright_license", "")).strip(),
            }
    return metadata


def list_images() -> list[dict]:
    metadata = load_image_metadata()
    images = []
    seen = set()
    for source, root in INPUT_ROOTS.items():
        if not root.exists():
            continue
        for path in sorted(root.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            image_id = path.stem
            if image_id in seen:
                continue
            seen.add(image_id)
            details = metadata.get(image_id, {})
            images.append({
                "id": image_id,
                "filename": path.name,
                "source": source,
                "url": f"/files/input/{source}/{path.name}",
                "fitzpatrick_skin_type": details.get("fitzpatrick_skin_type", ""),
                "diagnosis_1": details.get("diagnosis_1", ""),
                "image_type": details.get("image_type", ""),
                "copyright_license": details.get("copyright_license", ""),
            })
    return images


def image_path(record: dict) -> Path:
    source = record.get("source", "pilot")
    if source not in INPUT_ROOTS:
        raise ValueError(f"Origen de imagen desconocido: {source}")
    return require_within(
        INPUT_ROOTS[source] / record["filename"],
        (INPUT_ROOTS[source],),
        "Input image",
    )


def pool_summary(images: list[dict]) -> dict:
    counts = {
        fitzpatrick_type: sum(
            image.get("fitzpatrick_skin_type") == fitzpatrick_type for image in images
        )
        for fitzpatrick_type in FITZPATRICK_TYPES
    }
    return {
        "total_local_images": len(images),
        "eligible_fitzpatrick_images": sum(counts.values()),
        "fitzpatrick_counts": counts,
        "metadata_available": METADATA_CSV.is_file(),
        "full_image_directory": str(FITZPATRICK_INPUT_DIR.relative_to(REPO_ROOT)),
    }


def stratified_sample(images: list[dict], total: int, seed: int = 42) -> list[dict]:
    if total < len(FITZPATRICK_TYPES) or total % len(FITZPATRICK_TYPES) != 0:
        raise ValueError("La cantidad debe ser un múltiplo de 6 y al menos 6.")
    per_type = total // len(FITZPATRICK_TYPES)
    selected_by_type = {}
    for fitzpatrick_type in FITZPATRICK_TYPES:
        candidates = sorted(
            (
                image for image in images
                if image.get("fitzpatrick_skin_type") == fitzpatrick_type
            ),
            key=lambda image: image["id"],
        )
        if len(candidates) < per_type:
            raise ValueError(
                f"Fitzpatrick {fitzpatrick_type} solo tiene {len(candidates)} imágenes "
                f"locales; se necesitan {per_type}."
            )
        selected_by_type[fitzpatrick_type] = random.Random(
            f"{seed}:{fitzpatrick_type}"
        ).sample(candidates, per_type)
    return [
        selected_by_type[fitzpatrick_type][index]
        for index in range(per_type)
        for fitzpatrick_type in FITZPATRICK_TYPES
    ]


def initial_images(images: list[dict]) -> list[dict]:
    try:
        return stratified_sample(images, 12, 42)
    except ValueError:
        return images[:10]


def result_payload(
    model_id: str,
    image_id: str,
    image_path: Path,
    runtime: dict | None = None,
    *,
    original_url: str | None = None,
    image_metadata: dict | None = None,
) -> dict:
    output_dir = RESULTS_DIR / model_id / image_id
    lesion_path = output_dir / "lesion_mask.png"
    skin_path = output_dir / "clean_skin_mask.png"
    stats_path = output_dir / "colour_stats.json"
    runtime_path = output_dir / "runtime_metrics.json"
    if not lesion_path.is_file():
        return {
            "model_id": model_id,
            "image_id": image_id,
            "status": "mask_missing",
            "message": "El adaptador todavía no produjo lesion_mask.png.",
            "original_url": original_url or f"/files/input/{image_path.name}",
            "image_metadata": image_metadata or {},
        }

    image = Image.open(image_path).convert("RGB")
    lesion = normalize_binary_mask(Image.open(lesion_path), image.size)
    clean_skin, metadata = build_clean_skin_mask(image, lesion)
    save_binary_mask(lesion, lesion_path)
    save_binary_mask(clean_skin, skin_path)
    try:
        stats = measure_skin_colour(image, clean_skin)
        skin_colour = stats.to_dict()
        skin_colour_error = None
    except ValueError:
        skin_colour = None
        skin_colour_error = (
            "La predicción no dejó píxeles válidos de piel limpia; "
            "las estadísticas de color no están disponibles para este resultado."
        )
    stats_payload = {
        "postprocessing": metadata,
        "skin_colour": skin_colour,
        "skin_colour_available": skin_colour is not None,
        "skin_colour_error": skin_colour_error,
        "warning": "Fitzpatrick no se deduce únicamente a partir del color de la imagen.",
    }
    stats_path.write_text(json.dumps(stats_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if runtime is not None:
        runtime_path.write_text(json.dumps(runtime, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    elif runtime_path.is_file():
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    base = f"/files/result/{model_id}/{image_id}"
    return {
        "model_id": model_id,
        "image_id": image_id,
        "status": "ready",
        "original_url": original_url or f"/files/input/{image_path.name}",
        "lesion_mask_url": f"{base}/lesion_mask.png",
        "clean_skin_mask_url": f"{base}/clean_skin_mask.png",
        "stats": stats_payload,
        "runtime": runtime,
        "image_metadata": image_metadata or {},
    }


def _process_tree_rss_kb(root_pid: int) -> int:
    """Read aggregate resident memory for a Linux process and descendants."""
    processes: dict[int, tuple[int, int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text(encoding="utf-8")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        ppid = 0
        rss_kb = 0
        for line in status.splitlines():
            if line.startswith("PPid:"):
                ppid = int(line.split()[1])
            elif line.startswith("VmRSS:"):
                rss_kb = int(line.split()[1])
        processes[int(entry.name)] = (ppid, rss_kb)

    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _) in processes.items():
            if ppid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sum(processes.get(pid, (0, 0))[1] for pid in descendants)


def _runtime_metrics(start: float, peak_rss_kb: int, before: resource.struct_rusage) -> dict:
    wall_seconds = max(time.perf_counter() - start, 1e-9)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_seconds = max(
        0.0,
        (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime),
    )
    logical_cpus = os.cpu_count() or 1
    effective_cores = min(cpu_seconds / wall_seconds, float(logical_cpus))
    single_core_percent = effective_cores * 100.0
    system_capacity_percent = min(effective_cores / logical_cpus * 100.0, 100.0)
    return {
        "wall_seconds": round(wall_seconds, 4),
        "cpu_seconds": round(cpu_seconds, 4),
        "effective_cpu_cores": round(effective_cores, 4),
        "cpu_percent_single_core_equivalent": round(single_core_percent, 2),
        "cpu_percent_system_capacity": round(system_capacity_percent, 2),
        "logical_cpu_count": logical_cpus,
        "peak_ram_mb": round(peak_rss_kb / 1024.0, 2),
        "sampling_interval_seconds": 0.05,
    }


def benchmark_summaries(results: list[dict]) -> list[dict]:
    summaries = []
    model_ids = sorted({result["model_id"] for result in results})
    for model_id in model_ids:
        runtimes = [
            result["runtime"]
            for result in results
            if result["model_id"] == model_id
            and result.get("status") == "ready"
            and result.get("runtime")
        ]
        if not runtimes:
            continue
        summaries.append({
            "model_id": model_id,
            "images_timed": len(runtimes),
            "total_wall_seconds": round(sum(item["wall_seconds"] for item in runtimes), 4),
            "average_wall_seconds": round(
                sum(item["wall_seconds"] for item in runtimes) / len(runtimes), 4
            ),
            "average_cpu_percent_single_core_equivalent": round(
                sum(item["cpu_percent_single_core_equivalent"] for item in runtimes) / len(runtimes), 2
            ),
            "average_cpu_percent_system_capacity": round(
                sum(item["cpu_percent_system_capacity"] for item in runtimes) / len(runtimes), 2
            ),
            "average_effective_cpu_cores": round(
                sum(
                    item.get(
                        "effective_cpu_cores",
                        item["cpu_percent_single_core_equivalent"] / 100.0,
                    )
                    for item in runtimes
                ) / len(runtimes),
                4,
            ),
            "peak_ram_mb_max": round(max(item["peak_ram_mb"] for item in runtimes), 2),
        })
    return summaries


def run_adapter(model: dict, image_path: Path, lesion_path: Path, *, internal_warmup: int = 0) -> tuple[bool, str, dict | None]:
    """Run a configured adapter without invoking a shell."""
    template = model.get("adapter_command")
    if not template:
        return False, "El modelo está catalogado, pero todavía no tiene adaptador configurado.", None
    if not isinstance(template, list) or not all(isinstance(token, str) for token in template):
        return False, "adapter_command debe ser una lista JSON de argumentos.", None
    lesion_path.parent.mkdir(parents=True, exist_ok=True)
    adapter_metrics_path = lesion_path.with_suffix(lesion_path.suffix + ".adapter_metrics.json")
    adapter_metrics_path.unlink(missing_ok=True)
    replacements = {
        "{image}": str(image_path),
        "{lesion_mask}": str(lesion_path),
        "{repo_root}": str(REPO_ROOT),
        "{python}": sys.executable,
        "{conda}": os.environ.get("CONDA_EXE") or shutil.which("conda") or "conda",
        "{device}": os.environ.get("THESIS_DEVICE", "cpu"),
    }
    command = []
    for token in template:
        for placeholder, value in replacements.items():
            token = token.replace(placeholder, value)
        command.append(token)
    adapter_python = os.environ.get("THESIS_ADAPTER_PYTHON")
    if adapter_python:
        python_index = next(
            (
                index
                for index, token in enumerate(command)
                if Path(token).name in {"python", "python3"}
                or Path(token).name.startswith("python3.")
            ),
            -1,
        )
        if python_index < 0:
            return False, "THESIS_ADAPTER_PYTHON está definido, pero el comando no contiene python.", None
        command = [adapter_python, *command[python_index + 1 :]]
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    peak_rss_kb = 0
    try:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env={**os.environ, "THESIS_ADAPTER_METRICS_PATH": str(adapter_metrics_path), "THESIS_ADAPTER_WARMUP": str(internal_warmup)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        while True:
            peak_rss_kb = max(peak_rss_kb, _process_tree_rss_kb(process.pid))
            try:
                stdout, stderr = process.communicate(timeout=0.05)
                break
            except subprocess.TimeoutExpired:
                if time.perf_counter() - start > 900:
                    process.kill()
                    stdout, stderr = process.communicate()
                    metrics = _runtime_metrics(start, peak_rss_kb, before)
                    lesion_path.unlink(missing_ok=True)
                    return False, "El adaptador superó el límite de 900 segundos.", metrics
    except OSError as exc:
        return False, str(exc), None
    metrics = _runtime_metrics(start, peak_rss_kb, before)
    if adapter_metrics_path.is_file():
        try:
            metrics["adapter"] = json.loads(adapter_metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metrics["adapter"] = {"error": "invalid adapter timing sidecar"}
    if process.returncode != 0:
        detail = stderr.strip() or stdout.strip() or "sin detalle"
        lesion_path.unlink(missing_ok=True)
        return False, f"El adaptador terminó con código {process.returncode}: {detail[-800:]}", metrics
    if not lesion_path.is_file():
        return False, "El adaptador terminó, pero no creó lesion_mask.png.", metrics
    try:
        with Image.open(lesion_path) as candidate:
            candidate.verify()
    except (OSError, ValueError) as exc:
        lesion_path.unlink(missing_ok=True)
        return False, f"El adaptador creó una máscara inválida: {exc}", metrics
    return True, "ok", metrics


class ReviewHandler(SimpleHTTPRequestHandler):
    def _json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path: Path, root: Path) -> None:
        try:
            safe_path = require_within(path, (root,), "Requested file")
        except SystemExit:
            self.send_error(403)
            return
        if not safe_path.is_file():
            self.send_error(404)
            return
        content = safe_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(safe_path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)
        if path == "/api/state":
            images = list_images()
            self._json({
                "models": load_models(),
                "images": initial_images(images),
                "pool": pool_summary(images),
                "benchmark": benchmark_state(),
            })
            return
        if path == "/api/benchmark/state":
            self._json(benchmark_state())
            return
        if path == "/api/benchmark/runs":
            self._json({"runs": benchmark_runs()})
            return
        if path == "/api/benchmark/run":
            try: self._json(benchmark_run(query.get("run_id", [""])[0]))
            except (ValueError, FileNotFoundError) as exc: self._json({"error": str(exc)}, status=404)
            return
        if path == "/api/post05/state":
            try: self._json(post05_state())
            except (ValueError, FileNotFoundError, KeyError, OSError) as exc: self._json({"error": str(exc)}, status=409)
            return
        if path == "/api/post05/observations":
            self._json(post05_observations())
            return
        if path == "/api/post05/overlay":
            try:
                content = post05_overlay({key: values[0] for key, values in query.items() if values})
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except (ValueError, FileNotFoundError, KeyError, OSError) as exc:
                self._json({"error": str(exc)}, status=409)
            return
        if path == "/api/annotations/state":
            if not (ANNOTATION_PROJECT / "project.json").is_file(): self._json({"available": False, "message": "Inicialice el proyecto con scripts/benchmark/annotations.py init"})
            else:
                status = project_status(ANNOTATION_PROJECT)
                urls = {item["id"]: item["url"] for item in list_images()}
                for item in status["items"]: item["url"] = urls.get(item["image_id"])
                self._json({"available": True, **status})
            return
        if path.startswith("/files/input/"):
            relative = path.removeprefix("/files/input/")
            source, separator, filename = relative.partition("/")
            if separator and source in INPUT_ROOTS:
                self._serve_file(INPUT_ROOTS[source] / filename, INPUT_ROOTS[source])
            else:
                self._serve_file(PILOT_INPUT_DIR / relative, PILOT_INPUT_DIR)
            return
        if path.startswith("/files/result/"):
            self._serve_file(RESULTS_DIR / path.removeprefix("/files/result/"), RESULTS_DIR)
            return
        if path.startswith("/files/benchmark/"):
            self._serve_file(BENCHMARK_ARTIFACT_ROOT / "runs" / path.removeprefix("/files/benchmark/"), BENCHMARK_ARTIFACT_ROOT / "runs")
            return
        if path.startswith("/files/artifact/"):
            self._serve_file(BENCHMARK_ARTIFACT_ROOT / path.removeprefix("/files/artifact/"), BENCHMARK_ARTIFACT_ROOT)
            return
        if path == "/":
            path = "/index.html"
        self._serve_file(WEB_DIR / path.lstrip("/"), WEB_DIR)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path not in {"/api/infer", "/api/sample", "/api/annotations/save", "/api/post05/observation"}:
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            if path == "/api/post05/observation":
                self._json({"saved": save_post05_observation(request)}); return
            if path == "/api/annotations/save":
                if not (ANNOTATION_PROJECT / "project.json").is_file(): raise ValueError("Proyecto de anotación no inicializado")
                encoded = str(request.get("png_data_url", ""))
                if not encoded.startswith("data:image/png;base64,"): raise ValueError("Se requiere una máscara PNG")
                png = base64.b64decode(encoded.partition(",")[2], validate=True)
                record = save_mask_version(ANNOTATION_PROJECT, str(request.get("image_id", "")), str(request.get("role", "")), str(request.get("mask_type", "")), png, actor=str(request.get("actor", "")), note=str(request.get("note", "")))
                self._json({"saved": record, "status": project_status(ANNOTATION_PROJECT)}); return
            if path == "/api/sample":
                images = list_images()
                total = int(request.get("total", 12))
                seed = int(request.get("seed", 42))
                self._json({
                    "images": stratified_sample(images, total, seed),
                    "pool": pool_summary(images),
                    "total": total,
                    "per_type": total // len(FITZPATRICK_TYPES),
                    "seed": seed,
                })
                return
            requested_models = request.get("model_ids", [])
            requested_images = request.get("image_ids", [])
            models = {model["id"]: model for model in load_models()}
            images = {image["id"]: image for image in list_images()}
            if not requested_models or not requested_images:
                raise ValueError("Selecciona al menos un modelo y una imagen.")
            if len(requested_models) != len(set(requested_models)):
                raise ValueError("La selección contiene modelos duplicados.")

            results = []
            for model_id in requested_models:
                if model_id not in models:
                    raise ValueError(f"Modelo desconocido: {model_id}")
                for image_id in requested_images:
                    if image_id not in images:
                        raise ValueError(f"Imagen desconocida: {image_id}")
                    image_record = images[image_id]
                    current_image_path = image_path(image_record)
                    lesion_path = RESULTS_DIR / model_id / image_id / "lesion_mask.png"
                    runtime = None
                    if not lesion_path.is_file():
                        success, message, runtime = run_adapter(
                            models[model_id], current_image_path, lesion_path
                        )
                        if not success:
                            results.append({
                                "model_id": model_id,
                                "image_id": image_id,
                                "status": "adapter_unavailable",
                                "message": message,
                                "original_url": image_record["url"],
                                "runtime": runtime,
                                "image_metadata": image_record,
                            })
                            continue
                    results.append(result_payload(
                        model_id,
                        image_id,
                        current_image_path,
                        runtime,
                        original_url=image_record["url"],
                        image_metadata=image_record,
                    ))
            self._json({"results": results, "summaries": benchmark_summaries(results)})
        except (ValueError, json.JSONDecodeError, binascii.Error) as exc:
            self._json({"error": str(exc)}, status=400)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    PILOT_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    FITZPATRICK_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    print(f"Segmentation review: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
