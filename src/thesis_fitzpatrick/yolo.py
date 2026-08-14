"""Leakage-safe YOLOv3/Darknet-53 dataset and training utilities."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import re
import stat
import struct
import subprocess
from typing import Any

import cv2
import numpy as np

from .benchmark import atomic_write_bytes, atomic_write_json, content_hash, load_json, sha256_file
from .preprocessing import BBox


def bbox_from_mask(mask: np.ndarray) -> BBox:
    ys, xs = np.nonzero(mask > 0)
    if not len(xs):
        raise ValueError("máscara de entrenamiento vacía; no se genera una caja ficticia")
    return BBox(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def bbox_to_darknet(box: BBox, width: int, height: int) -> tuple[float, float, float, float]:
    return ((box.x0 + box.x1) / (2 * width), (box.y0 + box.y1) / (2 * height), box.width / width, box.height / height)


def darknet_to_bbox(values: tuple[float, float, float, float], width: int, height: int) -> BBox:
    cx, cy, bw, bh = values
    x0, y0 = round((cx - bw / 2) * width), round((cy - bh / 2) * height)
    x1, y1 = round((cx + bw / 2) * width), round((cy + bh / 2) * height)
    return BBox(max(0, x0), max(0, y0), min(width, x1), min(height, y1))


def perturb_bbox(box: BBox, width: int, height: int, fraction: float, rng: random.Random) -> BBox:
    dx, dy = round(box.width * fraction), round(box.height * fraction)
    x0 = max(0, box.x0 + rng.randint(-dx, dx))
    y0 = max(0, box.y0 + rng.randint(-dy, dy))
    x1 = min(width, box.x1 + rng.randint(-dx, dx))
    y1 = min(height, box.y1 + rng.randint(-dy, dy))
    return box if x1 <= x0 or y1 <= y0 else BBox(x0, y0, x1, y1)


def prepare_darknet_fold(manifest: dict[str, Any], data_root: Path, fold: dict[str, Any], output: Path, *, seed: int, perturbation_fraction: float = 0.0) -> dict[str, Any]:
    """Create Darknet labels only from this fold's authorized training IDs."""
    training_ids, validation_ids = set(fold["train_ids"]), set(fold["validation_ids"])
    if training_ids & validation_ids:
        raise ValueError("fuga: un ID aparece en training y validation")
    by_id = {item["image_id"]: item for item in manifest["items"]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "backup").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    lists: dict[str, list[str]] = {"train": [], "validation": []}
    failures = []
    for role, ids in (("train", training_ids), ("validation", validation_ids)):
        for image_id in sorted(ids):
            item = by_id[image_id]
            if not item.get("mask_paths"):
                failures.append({"image_id": image_id, "reason": "training mask missing"})
                continue
            image_path, mask_path = data_root / item["image_path"], data_root / item["mask_paths"][0]
            image, mask = cv2.imread(str(image_path)), cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None or image.shape[:2] != mask.shape:
                failures.append({"image_id": image_id, "reason": "invalid or dimension-mismatched image/mask"})
                continue
            try:
                box = bbox_from_mask(mask)
            except ValueError as exc:
                failures.append({"image_id": image_id, "reason": str(exc)})
                continue
            if role == "train" and perturbation_fraction:
                box = perturb_bbox(box, image.shape[1], image.shape[0], perturbation_fraction, rng)
            values = bbox_to_darknet(box, image.shape[1], image.shape[0])
            roundtrip = darknet_to_bbox(values, image.shape[1], image.shape[0])
            if max(abs(a - b) for a, b in zip(box.as_list(), roundtrip.as_list())) > 1:
                raise RuntimeError(f"round-trip YOLO excede un píxel para {image_id}")
            label_path = output / "labels" / f"{image_id}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(label_path, ("0 " + " ".join(f"{value:.10f}" for value in values) + "\n").encode())
            linked_image = output / "images" / f"{image_id}{image_path.suffix.lower()}"
            linked_image.parent.mkdir(parents=True, exist_ok=True)
            if not linked_image.exists(): linked_image.symlink_to(image_path.resolve())
            lists[role].append(str(linked_image.absolute()))
    for role, paths in lists.items():
        atomic_write_bytes(output / f"{role}.txt", ("\n".join(paths) + "\n").encode())
    atomic_write_bytes(output / "lesion.names", b"lesion\n")
    atomic_write_bytes(output / "lesion.data", (
        f"classes = 1\ntrain = {(output / 'train.txt').resolve()}\n"
        f"valid = {(output / 'validation.txt').resolve()}\n"
        f"names = {(output / 'lesion.names').resolve()}\n"
        f"backup = {(output / 'backup').resolve()}\n"
    ).encode())
    audit = {"schema_version": 1, "fold": fold["fold"], "seed": seed, "training_count": len(lists["train"]), "validation_count": len(lists["validation"]), "failures": failures, "ground_truth_policy": "labels generated exclusively from fold training/validation authorization; test is rejected by CLI"}
    atomic_write_json(output / "label_audit.json", audit)
    return audit


def patch_yolov3_cfg(source: Path, destination: Path, *, batch: int = 64, subdivisions: int = 16, width: int = 512, height: int = 512, classes: int = 1, learning_rate: float = 0.001, momentum: float = 0.9, weight_decay: float = 0.0005, max_batches: int = 6000) -> None:
    text = source.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^batch=\d+", f"batch={batch}", text, count=1)
    text = re.sub(r"(?m)^subdivisions=\d+", f"subdivisions={subdivisions}", text, count=1)
    text = re.sub(r"(?m)^width=\d+", f"width={width}", text, count=1)
    text = re.sub(r"(?m)^height=\d+", f"height={height}", text, count=1)
    text = re.sub(r"(?m)^learning_rate=.*$", f"learning_rate={learning_rate}", text, count=1)
    text = re.sub(r"(?m)^momentum=.*$", f"momentum={momentum}", text, count=1)
    text = re.sub(r"(?m)^decay=.*$", f"decay={weight_decay}", text, count=1)
    text = re.sub(r"(?m)^max_batches\s*=\s*\d+", f"max_batches={max_batches}", text, count=1)
    text = re.sub(r"(?m)^steps=.*$", f"steps={round(max_batches * 0.8)},{round(max_batches * 0.9)}", text, count=1)
    text = re.sub(r"(?m)^classes=\d+", f"classes={classes}", text)
    text = re.sub(r"(?m)^filters=255(?=\s*$)", f"filters={(classes + 5) * 3}", text)
    if text.count(f"classes={classes}") != 3 or text.count(f"filters={(classes + 5) * 3}") < 3:
        raise ValueError("La configuración no parece ser YOLOv3 completa con tres cabezas")
    atomic_write_bytes(destination, text.encode())


def _cfg_integer(path: Path, name: str) -> int:
    match = re.search(rf"(?m)^{re.escape(name)}\s*=\s*(\d+)\s*$", path.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"La configuración Darknet no declara {name}: {path}")
    return int(match.group(1))


def _darknet_data(path: Path) -> dict[str, str]:
    values = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "=" in raw and not raw.lstrip().startswith("#"):
            key, value = raw.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def validate_darknet_data(path: Path, fold: int) -> Path:
    values = _darknet_data(path)
    missing = {"classes", "train", "valid", "names", "backup"} - set(values)
    if missing:
        raise ValueError(f"lesion.data incompleto; faltan {sorted(missing)}")
    if values["classes"] != "1":
        raise ValueError("lesion.data debe declarar exactamente una clase")
    for name in ("train", "valid", "names"):
        artifact = Path(values[name])
        if not artifact.is_absolute() or not artifact.is_file():
            raise ValueError(f"La ruta {name} de lesion.data debe ser absoluta y existir: {artifact}")
    backup = Path(values["backup"])
    if not backup.is_absolute():
        raise ValueError("La ruta backup de lesion.data debe ser absoluta")
    expected = path.resolve().parent / "backup"
    if backup.resolve() != expected.resolve() or path.resolve().parent.name != f"fold-{fold}":
        raise ValueError(f"La ruta backup no corresponde al fold {fold}: {backup}")
    backup.mkdir(parents=True, exist_ok=True)
    if not backup.is_dir() or not os.access(backup, os.W_OK | os.X_OK):
        raise PermissionError(f"El directorio backup no existe o no es escribible: {backup}")
    return backup


def _regular_nonempty(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(path) from None
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        raise ValueError(f"Se esperaba un archivo regular no vacío: {path}")


def darknet_weights_iteration(path: Path, batch: int) -> int:
    """Read Darknet's header and return the completed batch count."""
    _regular_nonempty(path)
    with path.open("rb") as stream:
        header = stream.read(20)
    if len(header) < 16:
        raise ValueError(f"Checkpoint Darknet truncado: {path}")
    major, minor, _revision = struct.unpack("<3i", header[:12])
    seen_size = 8 if major * 10 + minor >= 2 and major < 1000 and minor < 1000 else 4
    if len(header) < 12 + seen_size:
        raise ValueError(f"Cabecera Darknet truncada: {path}")
    seen = struct.unpack("<Q" if seen_size == 8 else "<I", header[12:12 + seen_size])[0]
    if batch <= 0 or seen % batch:
        raise ValueError(f"Checkpoint Darknet tiene contador incompatible con batch={batch}: {path}")
    return seen // batch


def _git_commit(path: Path) -> str | None:
    completed = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=False, capture_output=True, text=True)
    return completed.stdout.strip() if completed.returncode == 0 else None


def _training_contract(darknet: Path, data_file: Path, cfg: Path, initial_weights: Path, fold: int) -> dict[str, Any]:
    data_artifacts = {name: {"path": str(Path(value).resolve()), "sha256": sha256_file(Path(value))} for name, value in _darknet_data(data_file).items() if name in {"train", "valid", "names"}}
    return {
        "fold": fold,
        "darknet_path": str(darknet.resolve()),
        "darknet_sha256": sha256_file(darknet),
        "data_path": str(data_file.resolve()),
        "data_sha256": sha256_file(data_file),
        "data_artifacts": data_artifacts,
        "cfg_path": str(cfg.resolve()),
        "cfg_sha256": sha256_file(cfg),
        "initial_weights_path": str(initial_weights.resolve()),
        "initial_weights_sha256": sha256_file(initial_weights),
        "max_batches": _cfg_integer(cfg, "max_batches"),
        "batch": _cfg_integer(cfg, "batch"),
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
    }


def _log_iteration(log: Path) -> int:
    iterations = [int(value) for value in re.findall(r"(?m)^(\d+):", log.read_text(encoding="utf-8", errors="replace"))]
    return max(iterations, default=0)


def validate_completed_training(state_path: Path, *, expected_fold: int | None = None) -> dict[str, Any]:
    state = load_json(state_path)
    contract = state.get("contract", {})
    fold = contract.get("fold")
    if state.get("schema_version") != 2 or state.get("status") != "completed" or not isinstance(fold, int):
        raise ValueError(f"Estado YOLO no completado o antiguo: {state_path}")
    if expected_fold is not None and fold != expected_fold:
        raise ValueError(f"Estado YOLO pertenece al fold {fold}, no al {expected_fold}")
    for name in ("darknet", "data", "cfg", "initial_weights"):
        artifact = Path(contract[f"{name}_path"])
        _regular_nonempty(artifact)
        if sha256_file(artifact) != contract[f"{name}_sha256"]:
            raise ValueError(f"Artefacto de entrenamiento YOLO modificado: {artifact}")
    for item in contract.get("data_artifacts", {}).values():
        artifact = Path(item["path"])
        _regular_nonempty(artifact)
        if sha256_file(artifact) != item["sha256"]:
            raise ValueError(f"Lista/nombres Darknet modificados: {artifact}")
    log = Path(state["log_path"])
    _regular_nonempty(log)
    text = log.read_text(encoding="utf-8", errors="replace")
    if re.search(r"Couldn't open file|cannot open file|fatal error", text, re.IGNORECASE):
        raise ValueError(f"El log Darknet contiene un error fatal: {log}")
    max_batches, batch = int(contract["max_batches"]), int(contract["batch"])
    if _log_iteration(log) < max_batches or int(state.get("observed_iteration", 0)) < max_batches:
        raise ValueError(f"Darknet no alcanzó max_batches={max_batches}")
    final_weights = Path(state["final_weights_path"])
    if final_weights.name != Path(contract["cfg_path"]).stem + "_final.weights":
        raise ValueError(f"Nombre de pesos finales inesperado: {final_weights}")
    if final_weights.parent.resolve() != validate_darknet_data(Path(contract["data_path"]), fold).resolve():
        raise ValueError("Los pesos finales no pertenecen al backup del fold")
    if darknet_weights_iteration(final_weights, batch) < max_batches:
        raise ValueError("Los pesos finales corresponden a un entrenamiento incompleto")
    if sha256_file(final_weights) != state.get("final_weights_sha256"):
        raise ValueError("El hash de los pesos finales cambió")
    return state


def _resume_checkpoint(path: Path, backup: Path, cfg: Path, contract: dict[str, Any]) -> int:
    if path.parent.resolve() != backup.resolve():
        raise ValueError(f"Checkpoint fuera del backup del fold: {path}")
    allowed = re.fullmatch(rf"{re.escape(cfg.stem)}_(\d+)\.weights", path.name) or path.name == f"{cfg.stem}.backup"
    if not allowed:
        raise ValueError(f"Nombre de checkpoint no reanudable: {path.name}")
    iteration = darknet_weights_iteration(path, int(contract["batch"]))
    if not 0 < iteration < int(contract["max_batches"]):
        raise ValueError(f"Iteración de checkpoint no reanudable: {iteration}")
    return iteration


def _automatic_resume(backup: Path, cfg: Path, contract: dict[str, Any], old_state: dict[str, Any] | None) -> Path | None:
    candidates = [
        path for path in backup.iterdir()
        if path.is_file() and (path.name == f"{cfg.stem}.backup" or re.fullmatch(rf"{re.escape(cfg.stem)}_\d+\.weights", path.name))
    ]
    if not candidates:
        return None
    if not old_state or old_state.get("contract") != contract:
        raise ValueError("Hay checkpoints, pero falta un contrato de entrenamiento compatible")
    valid = []
    for path in candidates:
        try:
            valid.append((_resume_checkpoint(path, backup, cfg, contract), path))
        except ValueError:
            continue
    if not valid:
        raise ValueError("No existe un checkpoint reanudable válido para este fold/configuración")
    return max(valid, key=lambda item: item[0])[1]


def run_darknet_training(darknet: Path, data_file: Path, cfg: Path, initial_weights: Path, output: Path, *, fold: int, resume_weights: Path | None = None) -> dict[str, Any]:
    for path in (darknet, data_file, cfg, initial_weights):
        _regular_nonempty(path)
    backup = validate_darknet_data(data_file, fold)
    output.mkdir(parents=True, exist_ok=True)
    state_path, log = output / "training_state.json", output / "darknet.log"
    contract = _training_contract(darknet, data_file, cfg, initial_weights, fold)
    old_state = load_json(state_path) if state_path.is_file() else None
    if resume_weights is None and old_state and old_state.get("status") == "completed" and old_state.get("contract") == contract:
        try:
            return {**validate_completed_training(state_path, expected_fold=fold), "reused": True}
        except (KeyError, OSError, ValueError):
            pass
    resume_weights = resume_weights or _automatic_resume(backup, cfg, contract, old_state)
    if resume_weights:
        if not old_state or old_state.get("contract") != contract:
            raise ValueError("La reanudación no coincide con el contrato del fold/configuración")
        resume_iteration = _resume_checkpoint(resume_weights, backup, cfg, contract)
    else:
        resume_iteration = 0
    final_weights = backup / f"{cfg.stem}_final.weights"
    if final_weights.exists():
        raise ValueError(f"Existen pesos finales sin un estado completed válido; limpie el fold antes de entrenar: {final_weights}")
    weights = resume_weights or initial_weights
    command = [str(darknet), "detector", "train", str(data_file), str(cfg), str(weights), "-dont_show", "-map"]
    state = {
        "schema_version": 2, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": command, "contract": contract, "resume": resume_weights is not None,
        "resume_weights_path": str(resume_weights.resolve()) if resume_weights else None,
        "resume_weights_sha256": sha256_file(resume_weights) if resume_weights else None,
        "resume_iteration": resume_iteration, "log_path": str(log.resolve()),
    }
    atomic_write_json(state_path, state)
    with log.open("wb") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    state.update({"returncode": completed.returncode, "observed_iteration": _log_iteration(log), "finished_utc": datetime.now(timezone.utc).isoformat()})
    try:
        if completed.returncode:
            raise RuntimeError(f"Darknet terminó con código {completed.returncode}")
        text = log.read_text(encoding="utf-8", errors="replace")
        if re.search(r"Couldn't open file|cannot open file|fatal error", text, re.IGNORECASE):
            raise RuntimeError("Darknet informó un error fatal en el log")
        if state["observed_iteration"] < contract["max_batches"]:
            raise RuntimeError(f"Darknet terminó en {state['observed_iteration']} de {contract['max_batches']} iteraciones")
        if darknet_weights_iteration(final_weights, contract["batch"]) < contract["max_batches"]:
            raise RuntimeError("Los pesos finales no alcanzaron max_batches")
        state.update({"status": "completed", "final_weights_path": str(final_weights.resolve()), "final_weights_bytes": final_weights.stat().st_size, "final_weights_sha256": sha256_file(final_weights)})
        atomic_write_json(state_path, state)
        validate_completed_training(state_path, expected_fold=fold)
        return state
    except (KeyError, OSError, ValueError, RuntimeError) as exc:
        state.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        atomic_write_json(state_path, state)
        raise RuntimeError(f"Entrenamiento YOLO inválido; revise {log}: {exc}") from exc


def validate_frozen_yolo(path: Path, expected_fold: int) -> dict[str, Any]:
    payload = load_json(path)
    identity = payload.pop("identity_hash", None)
    if payload.get("status") != "frozen" or payload.get("architecture") != "YOLOv3-Darknet53" or payload.get("fold") != expected_fold or content_hash(payload) != identity:
        raise ValueError(f"Manifest YOLO congelado inválido para fold {expected_fold}: {path}")
    payload["identity_hash"] = identity
    for name in ("cfg", "weights", "training_state", "validation_report"):
        artifact = Path(payload[f"{name}_path"])
        _regular_nonempty(artifact)
        if sha256_file(artifact) != payload[f"{name}_sha256"]:
            raise ValueError(f"Artefacto YOLO congelado modificado: {artifact}")
    state = validate_completed_training(Path(payload["training_state_path"]), expected_fold=expected_fold)
    if state["final_weights_sha256"] != payload["weights_sha256"]:
        raise ValueError(f"Peso congelado no coincide con el entrenamiento del fold {expected_fold}")
    return payload


def validate_frozen_yolo_set(root: Path) -> list[dict[str, Any]]:
    return [validate_frozen_yolo(root / f"fold-{fold}" / "frozen.json", fold) for fold in range(5)]


def box_iou(first: BBox, second: BBox) -> float:
    intersection = max(0, min(first.x1, second.x1) - max(first.x0, second.x0)) * max(0, min(first.y1, second.y1) - max(first.y0, second.y0))
    union = first.width * first.height + second.width * second.height - intersection
    return float(intersection / union) if union else 0.0


def select_validation_configuration(records: list[dict[str, Any]], confidence_candidates: list[float], nms_candidates: list[float], margin_candidates: list[float]) -> dict[str, Any]:
    """Select detector parameters from validation records containing raw boxes."""
    if not records: raise ValueError("No hay predicciones de validación")
    evaluations = []
    for confidence in confidence_candidates:
        for nms in nms_candidates:
            for margin in margin_candidates:
                ious, failures = [], 0
                for record in records:
                    width, height = record["image_size"]
                    candidates = [(BBox.from_list(item["bbox_xyxy_original"]), float(item["confidence"])) for item in record.get("detections", []) if float(item["confidence"]) >= confidence]
                    candidates.sort(key=lambda item: item[1], reverse=True)
                    kept = []
                    for box, score in candidates:
                        if all(box_iou(box, existing[0]) <= nms for existing in kept): kept.append((box, score))
                    if not kept:
                        failures += 1; ious.append(0.0); continue
                    selected = kept[0][0]
                    dx, dy = round(selected.width * margin), round(selected.height * margin)
                    expanded = BBox(max(0, selected.x0 - dx), max(0, selected.y0 - dy), min(width, selected.x1 + dx), min(height, selected.y1 + dy))
                    ious.append(box_iou(expanded, BBox.from_list(record["ground_truth_bbox_xyxy"])))
                evaluations.append({"confidence_threshold": confidence, "nms_threshold": nms, "margin_fraction": margin, "mean_iou": float(np.mean(ious)), "median_iou": float(np.median(ious)), "recall_iou_0.5": float(np.mean(np.asarray(ious) >= 0.5)), "failures": failures, "n": len(records)})
    selected = max(evaluations, key=lambda item: (item["mean_iou"], -item["failures"], -item["margin_fraction"]))
    return {"schema_version": 1, "selection_split": "validation", "selection_rule": "maximum mean box IoU, then fewer failures, then smaller margin", "selected": selected, "candidates": evaluations}


def collect_raw_validation_detections(cfg: Path, weights: Path, manifest: dict[str, Any], data_root: Path, validation_ids: set[str], *, input_size: tuple[int, int] = (512, 512), minimum_confidence: float = 0.01) -> list[dict[str, Any]]:
    if manifest.get("split") != "train": raise ValueError("La inferencia de selección solo acepta el manifest train y IDs del fold de validación")
    network = cv2.dnn.readNetFromDarknet(str(cfg), str(weights)); records = []
    if os.environ.get("THESIS_OPENCV_DNN_DEVICE") == "cuda":
        if not hasattr(cv2, "cuda") or cv2.cuda.getCudaEnabledDeviceCount() < 1:
            raise RuntimeError("OpenCV DNN CUDA was requested but this OpenCV build has no visible CUDA device")
        network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    for item in manifest["items"]:
        if item["image_id"] not in validation_ids: continue
        if not item.get("mask_paths"): raise ValueError(f"Falta máscara de validación autorizada: {item['image_id']}")
        image = cv2.imread(str(data_root / item["image_path"])); mask = cv2.imread(str(data_root / item["mask_paths"][0]), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or image.shape[:2] != mask.shape: raise ValueError(f"Imagen/máscara inválida: {item['image_id']}")
        height, width = image.shape[:2]; network.setInput(cv2.dnn.blobFromImage(image, 1 / 255.0, input_size, swapRB=True, crop=False))
        detections = []
        for output in network.forward(network.getUnconnectedOutLayersNames()):
            for row in output:
                confidence = float(row[4] * row[5])
                if confidence < minimum_confidence: continue
                cx, cy, bw, bh = float(row[0] * width), float(row[1] * height), float(row[2] * width), float(row[3] * height)
                x0, y0 = min(width - 1, max(0, int(round(cx - bw / 2)))), min(height - 1, max(0, int(round(cy - bh / 2))))
                x1, y1 = min(width, max(x0 + 1, int(round(cx + bw / 2)))), min(height, max(y0 + 1, int(round(cy + bh / 2))))
                detections.append({"bbox_xyxy_original": [x0, y0, x1, y1], "confidence": confidence})
        records.append({"image_id": item["image_id"], "image_size": [width, height], "ground_truth_bbox_xyxy": bbox_from_mask(mask).as_list(), "detections": detections})
    if len(records) != len(validation_ids): raise ValueError("No todos los IDs de validación fueron encontrados en el manifest")
    return records


def freeze_detector(cfg: Path, weights: Path, thresholds: dict[str, float], output: Path, validation_report: Path, *, fold: int, training_state: Path) -> dict[str, Any]:
    if not validation_report.is_file():
        raise FileNotFoundError("Falta informe de selección en validación")
    for key in ("confidence_threshold", "nms_threshold", "margin_fraction"):
        if key not in thresholds:
            raise ValueError(f"Falta umbral validado: {key}")
    validation = load_json(validation_report)
    selected = validation.get("selected", {})
    if any(float(selected.get(key, -1)) != float(value) for key, value in thresholds.items()):
        raise ValueError("Los umbrales a congelar no coinciden con la selección del informe de validación")
    state = validate_completed_training(training_state, expected_fold=fold)
    if Path(state["final_weights_path"]).resolve() != weights.resolve() or state["final_weights_sha256"] != sha256_file(weights) or Path(state["contract"]["cfg_path"]).resolve() != cfg.resolve() or state["contract"]["cfg_sha256"] != sha256_file(cfg):
        raise ValueError("La configuración/pesos a congelar no son los artefactos finales validados del fold")
    payload = {"schema_version": 2, "status": "frozen", "architecture": "YOLOv3-Darknet53", "fold": fold, "cfg_path": str(cfg.resolve()), "cfg_sha256": sha256_file(cfg), "weights_path": str(weights.resolve()), "weights_sha256": sha256_file(weights), "training_state_path": str(training_state.resolve()), "training_state_sha256": sha256_file(training_state), "validation_report_path": str(validation_report.resolve()), "validation_report_sha256": sha256_file(validation_report), "max_batches": state["contract"]["max_batches"], "observed_iteration": state["observed_iteration"], "thresholds": thresholds, "frozen_utc": datetime.now(timezone.utc).isoformat()}
    payload["identity_hash"] = content_hash(payload)
    atomic_write_json(output, payload)
    return payload
