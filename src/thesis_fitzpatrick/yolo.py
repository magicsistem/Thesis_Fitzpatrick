"""Leakage-safe YOLOv3/Darknet-53 dataset and training utilities."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
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


def run_darknet_training(darknet: Path, data_file: Path, cfg: Path, initial_weights: Path, output: Path, *, resume_weights: Path | None = None) -> dict[str, Any]:
    for path in (darknet, data_file, cfg, resume_weights or initial_weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=True)
    weights = resume_weights or initial_weights
    command = [str(darknet), "detector", "train", str(data_file), str(cfg), str(weights), "-dont_show", "-map"]
    state = {"schema_version": 1, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "command": command, "resume": resume_weights is not None, "cfg_sha256": sha256_file(cfg), "initial_weights_sha256": sha256_file(weights)}
    atomic_write_json(output / "training_state.json", state)
    log = output / "darknet.log"
    with log.open("ab") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    state.update({"status": "completed" if completed.returncode == 0 else "failed", "returncode": completed.returncode, "finished_utc": datetime.now(timezone.utc).isoformat()})
    atomic_write_json(output / "training_state.json", state)
    if completed.returncode:
        raise RuntimeError(f"Darknet terminó con código {completed.returncode}; revise {log}")
    return state


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


def freeze_detector(cfg: Path, weights: Path, thresholds: dict[str, float], output: Path, validation_report: Path) -> dict[str, Any]:
    if not validation_report.is_file():
        raise FileNotFoundError("Falta informe de selección en validación")
    for key in ("confidence_threshold", "nms_threshold", "margin_fraction"):
        if key not in thresholds:
            raise ValueError(f"Falta umbral validado: {key}")
    validation = load_json(validation_report)
    selected = validation.get("selected", {})
    if any(float(selected.get(key, -1)) != float(value) for key, value in thresholds.items()):
        raise ValueError("Los umbrales a congelar no coinciden con la selección del informe de validación")
    payload = {"schema_version": 1, "status": "frozen", "architecture": "YOLOv3-Darknet53", "cfg_path": str(cfg), "cfg_sha256": sha256_file(cfg), "weights_path": str(weights), "weights_sha256": sha256_file(weights), "thresholds": thresholds, "validation_report_sha256": sha256_file(validation_report), "frozen_utc": datetime.now(timezone.utc).isoformat()}
    payload["identity_hash"] = content_hash(payload)
    atomic_write_json(output, payload)
    return payload
