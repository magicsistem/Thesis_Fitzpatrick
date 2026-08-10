"""Dataset acquisition, integrity checks, and versioned manifest builders."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile
from typing import Any, Iterable

import cv2
import numpy as np

from .benchmark import atomic_write_json, sha256_file, validate_dataset_manifest


OFFICIAL_IMAPP_FILES = {
    "iaa_metrics_image.csv": "0bcf3c7d53ddb4f725400990c747381a",
    "iaa_metrics_pairwise.csv": "45829e731ead30901580abe608238543",
    "img_metadata.csv": "c27fdbcaab1e2cae48d6ad7edd3d64c4",
    "seg_metadata.csv": "aae21dec31b48e45786da70c4609e45f",
    "seg_metadata_multiannotator_subset.csv": "d8adefffa3ca1d7460aae5b7a5d1e661",
    "segs.zip": "c5143aed98283ab57de2c86ce50973c8",
    "test.csv": "92b3bf10a73a9bf1974acfad26bacfdc",
    "train.csv": "405995ddf4e912ed2bcf890f1c5c6eaf",
    "val.csv": "73ca2e9fa602e4bc18024adfe37c8b42",
}

OFFICIAL_ISIC2018_ARCHIVES = {
    "training_images": ("https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Training_Input.zip", "archives/ISIC2018_Task1-2_Training_Input.zip", 11165358566),
    "training_masks": ("https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Training_GroundTruth.zip", "archives/ISIC2018_Task1_Training_GroundTruth.zip", 27402895),
    "validation_images": ("https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Validation_Input.zip", "archives/ISIC2018_Task1-2_Validation_Input.zip", 239231159),
    "validation_masks": ("https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Validation_GroundTruth.zip", "archives/ISIC2018_Task1_Validation_GroundTruth.zip", 759706),
    "test_images": ("https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Test_Input.zip", "archives/ISIC2018_Task1-2_Test_Input.zip", 2370457338),
    "sealed_test_masks": ("https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Test_GroundTruth.zip", "sealed/archives/ISIC2018_Task1_Test_GroundTruth.zip", 9680101),
}


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_resumable(url: str, destination: Path, expected_digest: str | None = None, algorithm: str = "sha256", timeout: float = 120.0) -> dict[str, Any]:
    """Download to ``.partial`` and atomically publish only verified content."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        actual = file_digest(destination, algorithm)
        if expected_digest and actual.lower() != expected_digest.lower():
            raise ValueError(f"Checksum {algorithm} inválido para {destination.name}: {actual}")
        return {"path": str(destination), "bytes": destination.stat().st_size, algorithm: actual, "url": url, "reused": True}
    partial = destination.with_name(destination.name + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "Thesis-Fitzpatrick-reproducible-downloader/1.0", "Accept-Encoding": "identity"}
    if offset: headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - caller supplies registry URL
        if offset and response.status != 206:
            partial.unlink()
            offset = 0
        with partial.open("ab" if offset else "wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    actual = file_digest(partial, algorithm)
    if expected_digest and actual.lower() != expected_digest.lower():
        raise ValueError(f"Checksum {algorithm} inválido para {destination.name}: {actual}")
    partial.replace(destination)
    return {"path": str(destination), "bytes": destination.stat().st_size, algorithm: actual, "url": url}


def safe_extract_zip(archive: Path, destination: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        members = []
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Ruta insegura dentro de ZIP: {member.filename}")
            members.append(member.filename)
        bundle.extractall(destination)
    return members


def _image_files(root: Path) -> Iterable[Path]:
    for suffix in ("*.jpg", "*.jpeg", "*.png"):
        yield from root.rglob(suffix)


def _decode_image(path: Path, grayscale: bool = False) -> tuple[int, int]:
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    if value is None or value.size == 0:
        raise ValueError(f"Archivo de imagen corrupto o no decodificable: {path}")
    return value.shape[1], value.shape[0]


def build_isic2018_manifest(data_root: Path, split: str, *, include_checksums: bool = True, metadata_root: Path | None = None) -> dict[str, Any]:
    """Discover an imported official ISIC 2018 Task 1 split without moving it."""
    split_alias = {"train": "Training", "validation": "Validation", "test": "Test"}[split]
    candidates = [path for path in data_root.rglob("*") if path.is_dir() and split_alias.lower() in path.name.lower() and "groundtruth" not in path.name.lower() and "mask" not in path.name.lower()]
    image_paths = sorted({path for root in candidates for path in _image_files(root) if "groundtruth" not in str(path).lower() and "mask" not in path.name.lower()})
    by_id = {path.stem: path for path in image_paths if path.stem.startswith("ISIC_")}
    masks: dict[str, Path] = {}
    for path in _image_files(data_root):
        if split_alias.lower() in str(path.parent).lower() and ("segmentation" in path.stem.lower() or "groundtruth" in str(path).lower()):
            image_id = path.stem.replace("_segmentation", "")
            masks[image_id] = path
    items = []
    errors = []
    for image_id, image_path in sorted(by_id.items()):
        try:
            metadata_path = metadata_root / f"{image_id}.json" if metadata_root else None
            record = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path and metadata_path.is_file() else {}
            clinical = record.get("metadata", {}).get("clinical", {})
            patient_id = clinical.get("patient_id") or record.get("patient_id") or None
            lesion_id = clinical.get("lesion_id") or record.get("lesion_id") or None
            image_size = _decode_image(image_path)
            mask_path = masks.get(image_id)
            if split != "test" and mask_path is None:
                raise ValueError("falta la máscara oficial Task 1 para este split")
            if mask_path:
                mask_size = _decode_image(mask_path, grayscale=True)
                if mask_size != image_size:
                    raise ValueError(f"dimensiones imagen/máscara incompatibles: {image_size} != {mask_size}")
                mask_values = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if not set(np.unique(mask_values)).issubset({0, 255}):
                    raise ValueError("la máscara oficial debe ser binaria 0/255")
            item = {
                "image_id": image_id,
                "image_path": image_path.relative_to(data_root).as_posix(),
                "mask_paths": [mask_path.relative_to(data_root).as_posix()] if mask_path else [],
                "patient_id": patient_id,
                "lesion_id": lesion_id,
                "duplicate_group_id": lesion_id or image_id,
                "fitzpatrick": clinical.get("fitzpatrick_skin_type") or None,
                "image_size": list(image_size),
                "image_bytes": image_path.stat().st_size,
                "mask_bytes": mask_path.stat().st_size if mask_path else None,
                "copyright_license": record.get("copyright_license") or None,
                "attribution": record.get("attribution") or None,
                "metadata_path": metadata_path.relative_to(data_root).as_posix() if metadata_path and metadata_path.is_file() else None,
            }
            if include_checksums:
                item["image_sha256"] = sha256_file(image_path)
                item["mask_sha256"] = sha256_file(mask_path) if mask_path else None
            items.append(item)
        except ValueError as exc:
            errors.append({"image_id": image_id, "error": str(exc)})
    expected = {"train": 2594, "validation": 100, "test": 1000}[split]
    payload = {
        "schema_version": 1,
        "dataset_id": "isic2018_task1",
        "split": split,
        "source": "ISIC 2018 Challenge Task 1 official import",
        "expected_count": expected,
        "complete": len(items) == expected and not errors,
        "items": items,
        "integrity_errors": errors,
    }
    validate_dataset_manifest(payload)
    return payload


def verify_manifest_files(payload: dict[str, Any], data_root: Path, *, verify_checksums: bool = True) -> dict[str, Any]:
    audit = validate_dataset_manifest(payload, data_root=data_root, require_files=True)
    corrupt: list[dict[str, str]] = []
    for item in payload["items"]:
        for field, values in (("image", [item["image_path"]]), ("mask", item.get("mask_paths", []))):
            for relative in values:
                path = data_root / relative
                try:
                    _decode_image(path, grayscale=field == "mask")
                except ValueError as exc:
                    corrupt.append({"image_id": item["image_id"], "path": relative, "error": str(exc)})
        if verify_checksums and item.get("image_sha256"):
            actual = sha256_file(data_root / item["image_path"])
            if actual != item["image_sha256"]:
                corrupt.append({"image_id": item["image_id"], "path": item["image_path"], "error": "SHA-256 modificado"})
        if verify_checksums and item.get("mask_sha256"):
            expected = item["mask_sha256"]
            expected_values = expected if isinstance(expected, list) else [expected]
            mask_paths = item.get("mask_paths", [])
            if len(expected_values) != len(mask_paths):
                corrupt.append({"image_id": item["image_id"], "path": "mask_paths", "error": "Cantidad de SHA-256 de máscaras incompatible"})
            else:
                for relative, digest in zip(mask_paths, expected_values):
                    if digest and sha256_file(data_root / relative) != digest:
                        corrupt.append({"image_id": item["image_id"], "path": relative, "error": "SHA-256 de máscara modificado"})
    return {**audit, "corrupt_or_modified": corrupt, "passed": not corrupt}


def import_imapp_metadata(source_root: Path, data_root: Path) -> dict[str, Any]:
    """Verify official Zenodo v1.1 tables and build split manifests from their IDs."""
    verified = {}
    for filename, expected in OFFICIAL_IMAPP_FILES.items():
        path = source_root / filename
        if not path.is_file():
            continue
        actual = file_digest(path, "md5")
        if actual != expected:
            raise ValueError(f"MD5 oficial inválido para {filename}: {actual}")
        verified[filename] = actual
    required = {"img_metadata.csv", "seg_metadata.csv", "train.csv", "val.csv", "test.csv"}
    missing = sorted(required - set(verified))
    if missing:
        raise FileNotFoundError("Faltan archivos oficiales IMA++: " + ", ".join(missing))
    metadata = _csv_rows(source_root / "img_metadata.csv")
    id_field = next((key for key in metadata[0] if key.lower() in {"isic_id", "image_id", "image"}), None)
    if id_field is None:
        raise ValueError("img_metadata.csv no contiene una columna de identificador reconocida")
    image_meta = {row[id_field]: row for row in metadata}
    paths_by_name = {path.name: path for path in data_root.rglob("*") if path.is_file()}
    images_by_id = {path.stem: path for path in _image_files(data_root / "images")}
    segmentation_rows = _csv_rows(source_root / "seg_metadata.csv")
    masks_by_image: dict[str, list[tuple[str, Path, str]]] = {}
    invalid_masks = []
    for row in segmentation_rows:
        image_id = Path(row["ISIC_id"]).stem
        mask_path = paths_by_name.get(row["seg_filename"])
        if mask_path is None:
            continue
        actual_md5 = file_digest(mask_path, "md5")
        if actual_md5 != row["mask_md5"]:
            invalid_masks.append({"image_id": image_id, "path": str(mask_path), "expected_md5": row["mask_md5"], "actual_md5": actual_md5})
        masks_by_image.setdefault(image_id, []).append((row["annotator"], mask_path, sha256_file(mask_path)))
    if invalid_masks:
        raise ValueError(f"{len(invalid_masks)} máscaras IMA++ no coinciden con seg_metadata.csv: {invalid_masks[0]}")
    manifests = {}
    expected_counts = {"train": 1675, "validation": 240, "test": 479}
    for split, filename in (("train", "train.csv"), ("validation", "val.csv"), ("test", "test.csv")):
        rows = _csv_rows(source_root / filename)
        split_id_field = next((key for key in rows[0] if key.lower() in {"isic_id", "image_id", "image"}), None)
        if split_id_field is None:
            raise ValueError(f"{filename} no contiene identificadores de imagen")
        items = []
        for row in rows:
            image_id = Path(row[split_id_field]).stem
            image = images_by_id.get(image_id)
            annotated_masks = masks_by_image.get(image_id, [])
            staple = [path for annotator, path, _ in annotated_masks if annotator == "ST"]
            majority = [path for annotator, path, _ in annotated_masks if annotator == "MV"]
            individuals = [path for annotator, path, _ in annotated_masks if annotator not in {"ST", "MV"}]
            masks = staple + majority + sorted(individuals)
            digest_by_path = {path: digest for _, path, digest in annotated_masks}
            details = image_meta.get(image_id, {})
            image_size = _decode_image(image) if image else None
            if image_size:
                for mask in masks:
                    if _decode_image(mask, grayscale=True) != image_size:
                        raise ValueError(f"Dimensiones imagen/máscara IMA++ incompatibles: {image_id} / {mask.name}")
            items.append({
                "image_id": image_id,
                "image_path": image.relative_to(data_root).as_posix() if image else f"images/{image_id}.jpg",
                "mask_paths": [path.relative_to(data_root).as_posix() for path in masks],
                "image_sha256": sha256_file(image) if image else None,
                "mask_sha256": [digest_by_path[path] for path in masks],
                "image_bytes": image.stat().st_size if image else None,
                "mask_bytes": [path.stat().st_size for path in masks],
                "image_size": list(image_size) if image_size else None,
                "staple_mask_path": staple[0].relative_to(data_root).as_posix() if staple else None,
                "majority_mask_path": majority[0].relative_to(data_root).as_posix() if majority else None,
                "individual_mask_count": len(individuals),
                "patient_id": details.get("patient_id") or None,
                "lesion_id": details.get("lesion_id") or None,
                "duplicate_group_id": image_id,
                "fitzpatrick": details.get("fitzpatrick_skin_type") or None,
                "reference_policy": "STAPLE_primary_majority_secondary_individuals_reported",
            })
        manifests[split] = {"schema_version": 1, "dataset_id": "imapp", "split": split, "expected_count": expected_counts[split], "complete": len(items) == expected_counts[split] and all((data_root / item["image_path"]).is_file() and item["staple_mask_path"] and item["majority_mask_path"] for item in items), "items": items}
    return {"verified_md5": verified, "segmentation_masks_verified": sum(len(value) for value in masks_by_image.values()), "manifests": manifests}


def build_fitzpatrick_manifest(images_root: Path, metadata_csv: Path, data_root: Path) -> dict[str, Any]:
    rows = _csv_rows(metadata_csv)
    id_key = next((key for key in rows[0] if key.lower() in {"isic_id", "image_id"}), None)
    fitz_key = next((key for key in rows[0] if "fitzpatrick" in key.lower()), None)
    if not id_key or not fitz_key: raise ValueError("El CSV requiere image_id/isic_id y Fitzpatrick")
    metadata = {row[id_key]: row for row in rows}
    hashes: dict[str, list[str]] = {}
    items = []
    for path in sorted(_image_files(images_root)):
        image_id = path.stem; digest = sha256_file(path); hashes.setdefault(digest, []).append(image_id)
        row = metadata.get(image_id, {})
        fitzpatrick = str(row.get(fitz_key, "")).strip() or None
        if fitzpatrick not in {None, "I", "II", "III", "IV", "V", "VI"}: raise ValueError(f"Fitzpatrick inválido para {image_id}: {fitzpatrick}")
        items.append({"image_id": image_id, "image_path": path.relative_to(data_root).as_posix(), "mask_paths": [], "fov_mask_path": None, "patient_id": row.get("patient_id") or None, "lesion_id": row.get("lesion_id") or None, "duplicate_group_id": digest, "fitzpatrick": fitzpatrick, "image_sha256": digest, "image_size": list(_decode_image(path)), "adjudication_status": "pending"})
    duplicate_groups = [ids for ids in hashes.values() if len(ids) > 1]
    return {"schema_version": 1, "dataset_id": "fitzpatrick_external", "split": "external", "items": items, "overlap_audit": {"sha256_duplicate_groups": duplicate_groups, "passed": not duplicate_groups}, "ground_truth_available": False}


def build_novice_manifest(masks_root: Path, data_root: Path, *, mapping_csv: Path | None = None, metadata_csv: Path | None = None) -> dict[str, Any]:
    mapping_csv = mapping_csv or data_root / "supplements" / "Masks_to_IsicId_mapping.csv"
    metadata_csv = metadata_csv or data_root / "metadata.csv"
    mapping = {row["Filename"]: row["ISIC_ID"] for row in _csv_rows(mapping_csv)} if mapping_csv.is_file() else {}
    metadata = {row["isic_id"]: row for row in _csv_rows(metadata_csv)} if metadata_csv.is_file() else {}
    items = []
    for path in sorted(_image_files(masks_root)):
        image_id = mapping.get(path.name, path.stem.removesuffix("_UNET").removesuffix("_segmentation"))
        image_path = data_root / "images" / f"{image_id}.jpg"
        if not image_path.is_file(): raise FileNotFoundError(f"Falta imagen oficial Novice para {path.name}: {image_path}")
        if _decode_image(path, grayscale=True) != _decode_image(image_path): raise ValueError(f"Dimensiones imagen/máscara Novice incompatibles: {image_id}")
        details = metadata.get(image_id, {})
        items.append({"image_id": image_id, "image_path": image_path.relative_to(data_root).as_posix(), "mask_paths": [path.relative_to(data_root).as_posix()], "image_sha256": sha256_file(image_path), "mask_sha256": sha256_file(path), "image_bytes": image_path.stat().st_size, "mask_bytes": path.stat().st_size, "patient_id": details.get("patient_id") or None, "lesion_id": details.get("lesion_id") or None, "duplicate_group_id": details.get("lesion_id") or image_id, "fitzpatrick": details.get("fitzpatrick_skin_type") or None, "annotation_role": "novice_noisy_auxiliary_only"})
    return {"schema_version": 1, "dataset_id": "isic2018_novice_masks", "split": "auxiliary", "enabled": False, "complete": len(items) == len(mapping) == 11720, "license": "CC-BY-NC", "warning": "NO APTO PARA ELEGIR EL MODELO PRINCIPAL", "items": items}


def majority_consensus(masks: list[np.ndarray]) -> np.ndarray:
    if not masks or any(mask.shape != masks[0].shape for mask in masks): raise ValueError("Se requieren máscaras no vacías de igual tamaño")
    return (np.mean(np.stack([mask > 0 for mask in masks]), axis=0) >= 0.5).astype(np.uint8)


def staple_consensus(masks: list[np.ndarray], *, iterations: int = 50, tolerance: float = 1e-6) -> np.ndarray:
    """Deterministic binary STAPLE EM without adding a heavyweight dependency."""
    if len(masks) < 2 or any(mask.shape != masks[0].shape for mask in masks): raise ValueError("STAPLE requiere al menos dos máscaras del mismo tamaño")
    observations = np.stack([mask > 0 for mask in masks]).astype(float).reshape(len(masks), -1)
    probability = np.clip(observations.mean(axis=0), 1e-5, 1 - 1e-5)
    sensitivity = np.full(len(masks), 0.999); specificity = np.full(len(masks), 0.999)
    prevalence = float(probability.mean())
    for _ in range(iterations):
        previous = probability.copy()
        log_foreground = np.log(max(prevalence, 1e-9)) + (observations * np.log(sensitivity[:, None]) + (1 - observations) * np.log(1 - sensitivity[:, None])).sum(axis=0)
        log_background = np.log(max(1 - prevalence, 1e-9)) + ((1 - observations) * np.log(specificity[:, None]) + observations * np.log(1 - specificity[:, None])).sum(axis=0)
        maximum = np.maximum(log_foreground, log_background)
        probability = np.exp(log_foreground - maximum) / (np.exp(log_foreground - maximum) + np.exp(log_background - maximum))
        sensitivity = np.clip((observations * probability).sum(axis=1) / max(probability.sum(), 1e-9), 1e-5, 1 - 1e-5)
        specificity = np.clip(((1 - observations) * (1 - probability)).sum(axis=1) / max((1 - probability).sum(), 1e-9), 1e-5, 1 - 1e-5)
        prevalence = float(np.clip(probability.mean(), 1e-5, 1 - 1e-5))
        if float(np.max(np.abs(probability - previous))) < tolerance: break
    return (probability.reshape(masks[0].shape) >= 0.5).astype(np.uint8)


def audit_manifest_overlap(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    collisions = []
    seen: dict[tuple[str, str], tuple[str, str]] = {}
    for manifest in manifests:
        label = f"{manifest['dataset_id']}:{manifest['split']}"
        for item in manifest["items"]:
            identities = [(key, str(item[key])) for key in ("image_sha256", "patient_id", "lesion_id", "duplicate_group_id") if item.get(key) not in (None, "")]
            for identity in identities:
                previous = seen.get(identity)
                if previous and previous[0] != label: collisions.append({"identity_type": identity[0], "identity": identity[1], "first_dataset_split": previous[0], "first_image_id": previous[1], "second_dataset_split": label, "second_image_id": item["image_id"]})
                else: seen[identity] = (label, item["image_id"])
    return {"passed": not collisions, "collisions": collisions, "manifests": [f"{item['dataset_id']}:{item['split']}" for item in manifests]}


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    validate_dataset_manifest(payload)
    atomic_write_json(path, payload)
