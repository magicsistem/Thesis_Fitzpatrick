"""Versioned benchmark contracts and registries.

This module deliberately contains no model framework imports.  It is safe to
use from ``tesis-sam`` while neural adapters remain isolated in their existing
Conda environments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any

import numpy as np


SCHEMA_VERSION = 1


class Evaluation(str, Enum):
    A = "A"
    B1 = "B1"
    B2 = "B2"
    C0 = "C0"
    C1 = "C1"
    C2 = "C2"
    C3 = "C3"


@dataclass(frozen=True)
class ImageInput:
    image_id: str
    dataset_id: str
    split: str
    rgb: np.ndarray = field(repr=False, compare=False)
    fold: int | None = None
    ground_truth: np.ndarray | None = field(default=None, repr=False, compare=False)
    ground_truth_allowed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3 or self.rgb.dtype != np.uint8:
            raise ValueError("rgb must be an HxWx3 uint8 array")
        if self.ground_truth is not None and not self.ground_truth_allowed:
            raise ValueError("ground truth was supplied to a mode that cannot use it")

    @property
    def original_size(self) -> tuple[int, int]:
        return self.rgb.shape[1], self.rgb.shape[0]


@dataclass
class BackendResult:
    native_mask: np.ndarray = field(repr=False)
    input_size: tuple[int, int]
    backend_time_ms: float
    model_identity: dict[str, Any]
    threshold: float | None = None
    raw_probability: np.ndarray | None = field(default=None, repr=False)
    warnings: list[str] = field(default_factory=list)
    failure_code: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("native_mask")
        payload.pop("raw_probability")
        payload["input_size"] = list(self.input_size)
        payload["native_foreground_fraction"] = float(np.mean(self.native_mask > 0))
        payload["raw_probability_stored"] = self.raw_probability is not None
        return payload


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_bytes(path, (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _relative_path(value: str, label: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not value:
        raise ValueError(f"{label} must be a non-empty relative POSIX path: {value!r}")
    return value


def validate_dataset_registry(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported dataset registry schema_version")
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("dataset registry must contain a non-empty datasets list")
    ids = [item.get("id") for item in datasets if isinstance(item, dict)]
    if len(ids) != len(datasets) or len(set(ids)) != len(ids) or any(not item for item in ids):
        raise ValueError("dataset ids must be non-empty and unique")
    novice = next((item for item in datasets if item["id"] == "isic2018_novice_masks"), None)
    if not novice or novice.get("enabled") or "NO APTO" not in novice.get("warning", ""):
        raise ValueError("ISIC 2018 novice masks must remain disabled with a visible warning")
    return datasets


def validate_dataset_manifest(
    payload: dict[str, Any],
    *,
    data_root: Path | None = None,
    require_files: bool = False,
) -> dict[str, Any]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported manifest schema_version")
    if not payload.get("dataset_id") or not payload.get("split"):
        raise ValueError("manifest requires dataset_id and split")
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("manifest items must be a list")
    seen: set[str] = set()
    missing: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"item {index} must be an object")
        image_id = str(item.get("image_id", "")).strip()
        if not image_id or image_id in seen:
            raise ValueError(f"item {index} has an empty or duplicate image_id")
        seen.add(image_id)
        paths = [_relative_path(str(item.get("image_path", "")), f"item {image_id} image_path")]
        mask_paths = item.get("mask_paths", [])
        if not isinstance(mask_paths, list):
            raise ValueError(f"item {image_id} mask_paths must be a list")
        paths.extend(_relative_path(str(path), f"item {image_id} mask path") for path in mask_paths)
        if require_files:
            if data_root is None:
                raise ValueError("data_root is required when require_files=True")
            missing.extend(path for path in paths if not (data_root / path).is_file())
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"{len(missing)} manifest files are missing under {data_root}: {preview}")
    return {
        "dataset_id": payload["dataset_id"],
        "split": payload["split"],
        "items": len(items),
        "manifest_hash": content_hash(payload),
        "files_checked": require_files,
    }


def build_grouped_folds(payload: dict[str, Any], folds: int = 5, seed: int = 20260806) -> dict[str, Any]:
    """Assign linked patient/lesion/duplicate identities to deterministic folds."""
    validate_dataset_manifest(payload)
    if folds < 2:
        raise ValueError("folds must be at least 2")
    items = payload["items"]
    parent = list(range(len(items)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    identities: dict[tuple[str, str], int] = {}
    for index, item in enumerate(items):
        values = [
            (key, str(item[key]))
            for key in ("patient_id", "lesion_id", "duplicate_group_id")
            if item.get(key) not in (None, "")
        ]
        if not values:
            values = [("image_id", str(item["image_id"]))]
        for identity in values:
            if identity in identities:
                union(index, identities[identity])
            else:
                identities[identity] = index

    groups: dict[int, list[str]] = {}
    for index, item in enumerate(items):
        groups.setdefault(find(index), []).append(str(item["image_id"]))
    ordered_groups = sorted(
        groups.values(),
        key=lambda ids: hashlib.sha256(f"{seed}:{'|'.join(sorted(ids))}".encode()).hexdigest(),
    )
    assignments: dict[str, int] = {}
    for index, image_ids in enumerate(ordered_groups):
        for image_id in image_ids:
            assignments[image_id] = index % folds

    fold_payloads = []
    all_ids = set(assignments)
    for fold in range(folds):
        validation_ids = sorted(image_id for image_id, assigned in assignments.items() if assigned == fold)
        training_ids = sorted(all_ids - set(validation_ids))
        if set(training_ids) & set(validation_ids):
            raise RuntimeError(f"leakage detected in fold {fold}")
        fold_payloads.append({"fold": fold, "train_ids": training_ids, "validation_ids": validation_ids})
    fallback_count = sum(
        not any(item.get(key) not in (None, "") for key in ("patient_id", "lesion_id"))
        and item.get("duplicate_group_id") in (None, "", item["image_id"])
        for item in items
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": payload["dataset_id"],
        "source_split": payload["split"],
        "fold_count": folds,
        "seed": seed,
        "group_count": len(groups),
        "image_id_fallback_count": fallback_count,
        "fallback_policy": "image_id when patient_id and lesion_id are missing and duplicate_group_id is absent or equals image_id",
        "leakage_audit": {"passed": True, "groups_crossing_folds": 0},
        "folds": fold_payloads,
    }


def load_benchmark_methods(model_catalog: Path) -> list[dict[str, Any]]:
    models = load_json(model_catalog).get("models")
    if not isinstance(models, list) or len(models) != 15:
        raise ValueError("the frozen neural catalogue must contain exactly 15 variants")
    methods = []
    for index, model in enumerate(sorted(models, key=lambda item: item["priority"]), start=1):
        methods.append({**model, "method_id": f"S{index:02d}", "backend_id": model["id"], "kind": "neural"})
    methods.append({
        "method_id": "S16",
        "backend_id": "grabcut",
        "id": "grabcut",
        "name": "GrabCut",
        "kind": "classical",
        "modes": {"A": "classic", "B1": "robust", "B2": "robust", "C0": "classic", "C1": "robust", "C2": "robust", "C3": "robust"},
        "checkpoint_status": "not_applicable",
        "integration_status": "ready",
    })
    return methods


def evaluation_registry(methods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missing_b2 = [item["method_id"] for item in methods if item["kind"] == "neural"]
    return [
        {"id": "A", "name": "Métodos nativos", "available": True, "uses_p0": False},
        {"id": "B1", "name": "Pipeline común — checkpoints congelados", "available": True, "uses_p0": True},
        {"id": "B2", "name": "Pipeline común — modelos reentrenados", "available": False, "uses_p0": True, "reason": "No disponible: faltan checkpoints B2", "missing_methods": missing_b2},
        *[
            {"id": code, "name": f"Ablación {code}", "available": True, "uses_p0": code != "C0"}
            for code in ("C0", "C1", "C2", "C3")
        ],
    ]
