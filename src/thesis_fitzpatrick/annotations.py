"""Versioned two-reader lesion/FOV annotation projects with adjudication."""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .benchmark import atomic_write_bytes, atomic_write_json, load_json, sha256_file


ROLES = ("annotator_1", "annotator_2", "adjudicator")
MASK_TYPES = ("lesion", "fov")


def initialize_project(root: Path, dataset_manifest: dict[str, Any]) -> dict[str, Any]:
    if root.exists():
        raise FileExistsError(f"No se sobrescribe el proyecto existente: {root}")
    payload = {
        "schema_version": 1,
        "dataset_id": dataset_manifest["dataset_id"],
        "source_manifest_hash": __import__("hashlib").sha256(json.dumps(dataset_manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "items": [{"image_id": item["image_id"], "image_path": item["image_path"], "image_size": item.get("image_size"), "fitzpatrick": item.get("fitzpatrick"), "patient_id": item.get("patient_id"), "lesion_id": item.get("lesion_id"), "versions": []} for item in dataset_manifest["items"]],
    }
    atomic_write_json(root / "project.json", payload)
    atomic_write_bytes(root / "audit.jsonl", b"")
    return payload


def project_status(root: Path) -> dict[str, Any]:
    project = load_json(root / "project.json")
    counts = {"pending": 0, "double_annotated": 0, "complete": 0}
    items = []
    for item in project["items"]:
        roles = {version["role"] for version in item.get("versions", []) if version["mask_type"] == "lesion"}
        status = "complete" if "adjudicator" in roles else "double_annotated" if {"annotator_1", "annotator_2"} <= roles else "pending"
        counts[status] += 1
        items.append({"image_id": item["image_id"], "status": status, "roles": sorted(roles), "fitzpatrick": item.get("fitzpatrick"), "image_path": item["image_path"]})
    return {"dataset_id": project["dataset_id"], "counts": counts, "items": items}


def save_mask_version(root: Path, image_id: str, role: str, mask_type: str, png: bytes, *, actor: str, note: str = "") -> dict[str, Any]:
    if role not in ROLES or mask_type not in MASK_TYPES:
        raise ValueError("role o mask_type inválido")
    if not actor.strip():
        raise ValueError("actor es obligatorio para trazabilidad")
    project = load_json(root / "project.json")
    item = next((value for value in project["items"] if value["image_id"] == image_id), None)
    if item is None:
        raise ValueError(f"image_id no registrado: {image_id}")
    try:
        with Image.open(io.BytesIO(png)) as candidate:
            mask = np.asarray(candidate.convert("L"))
    except OSError as exc:
        raise ValueError("PNG de máscara inválido") from exc
    binary = ((mask > 127) * 255).astype(np.uint8)
    if item.get("image_size") and [binary.shape[1], binary.shape[0]] != item["image_size"]:
        raise ValueError(f"Dimensiones de máscara {binary.shape[1]}x{binary.shape[0]} no coinciden con la imagen {item['image_size']}")
    if not np.any(binary) and mask_type == "lesion":
        raise ValueError("La máscara de lesión está vacía; confirme externamente un caso verdaderamente vacío")
    versions = [value for value in item.get("versions", []) if value["role"] == role and value["mask_type"] == mask_type]
    version = 1 + max((int(value["version"]) for value in versions), default=0)
    relative = Path("masks") / image_id / role / f"{mask_type}.v{version:04d}.png"
    stream = io.BytesIO()
    Image.fromarray(binary, mode="L").save(stream, format="PNG")
    atomic_write_bytes(root / relative, stream.getvalue())
    record = {"role": role, "mask_type": mask_type, "version": version, "path": relative.as_posix(), "sha256": sha256_file(root / relative), "actor": actor.strip(), "note": note.strip(), "created_utc": datetime.now(timezone.utc).isoformat()}
    item.setdefault("versions", []).append(record)
    atomic_write_json(root / "project.json", project)
    event = {"event": "mask_saved", "image_id": image_id, **record}
    with (root / "audit.jsonl").open("a", encoding="utf-8") as audit:
        audit.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    return record


def sync_project(root: Path, dataset_manifest: dict[str, Any]) -> dict[str, Any]:
    project = load_json(root / "project.json"); source = {item["image_id"]: item for item in dataset_manifest["items"]}
    if set(source) != {item["image_id"] for item in project["items"]}: raise ValueError("El manifest sincronizado no contiene exactamente los mismos IDs")
    for item in project["items"]:
        current = source[item["image_id"]]
        for key in ("image_path", "image_size", "fitzpatrick", "patient_id", "lesion_id"): item[key] = current.get(key)
    project["source_manifest_hash"] = __import__("hashlib").sha256(json.dumps(dataset_manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    project["synced_utc"] = datetime.now(timezone.utc).isoformat(); atomic_write_json(root / "project.json", project)
    return {"items": len(project["items"]), "synced_utc": project["synced_utc"]}


def export_consensus_manifest(root: Path, output: Path) -> dict[str, Any]:
    project = load_json(root / "project.json")
    common_root = root.parents[1] if root.parent.name == "interim" else root.parent
    items = []
    missing = []
    for item in project["items"]:
        masks = {}
        for mask_type in MASK_TYPES:
            candidates = [value for value in item.get("versions", []) if value["mask_type"] == mask_type]
            candidates.sort(key=lambda value: (value["role"] == "adjudicator", value["version"]), reverse=True)
            if candidates:
                masks[mask_type] = (root / candidates[0]["path"]).relative_to(common_root).as_posix()
        if "lesion" not in masks:
            missing.append(item["image_id"])
        image_path = item["image_path"]
        if not (common_root / image_path).is_file() and (common_root / "raw" / image_path).is_file(): image_path = (Path("raw") / image_path).as_posix()
        items.append({"image_id": item["image_id"], "image_path": image_path, "mask_paths": [masks["lesion"]] if "lesion" in masks else [], "fov_mask_path": masks.get("fov"), "patient_id": item.get("patient_id"), "lesion_id": item.get("lesion_id"), "duplicate_group_id": item.get("lesion_id") or item["image_id"], "fitzpatrick": item.get("fitzpatrick"), "adjudication_status": "complete" if "lesion" in masks else "pending"})
    payload = {"schema_version": 1, "dataset_id": project["dataset_id"], "split": "external", "data_root": str(common_root), "annotation_project": str(root), "complete": not missing, "missing_lesion_masks": missing, "items": items}
    atomic_write_json(output, payload)
    return payload
