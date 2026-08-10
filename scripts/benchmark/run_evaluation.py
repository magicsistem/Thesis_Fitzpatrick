#!/usr/bin/env python3
"""Run native A or common-pipeline B1 without conflating B2."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from thesis_fitzpatrick.benchmark import (  # noqa: E402
    BackendResult,
    ImageInput,
    atomic_write_bytes,
    atomic_write_json,
    content_hash,
    evaluation_registry,
    load_benchmark_methods,
    load_json,
    sha256_file,
    validate_dataset_manifest,
    validate_dataset_registry,
)
from thesis_fitzpatrick.grabcut import common_postprocess, grabcut_classic, grabcut_robust  # noqa: E402
from thesis_fitzpatrick.metrics import segmentation_metrics  # noqa: E402
from thesis_fitzpatrick.preprocessing import atomic_write_png, detector_from_config, run_p0  # noqa: E402
from thesis_fitzpatrick.reporting import write_report  # noqa: E402
import serve_segmentation_review as review  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, choices=("A", "B1", "B2"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--models", default="all", help="all, S01,...,S16, or catalogue backend ids")
    parser.add_argument("--manifest", type=Path, help="Versioned dataset manifest")
    parser.add_argument("--data-root", type=Path, help="Root used by relative manifest paths")
    parser.add_argument("--image", action="append", type=Path, default=[], help="Explicit image for an ad-hoc smoke run; repeatable")
    parser.add_argument("--image-ids", help="Comma-separated manifest IDs; never allowed for sealed test")
    parser.add_argument("--limit", type=int, help="Limit a development/smoke run")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "benchmark" / "default.json")
    parser.add_argument("--b2-config", type=Path, help="Frozen B2 checkpoint registry; required for B2")
    parser.add_argument("--artifact-root", type=Path, help="Override config artifact_root")
    parser.add_argument("--run-id", help="Stable local run id; must not already exist")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse-p0-cache", action="store_true", help="Reuse validated P0 artifacts instead of measuring P0 again")
    parser.add_argument("--warmup", type=int, default=0, help="Unmeasured adapter process warm-ups")
    parser.add_argument("--repetitions", type=int, default=1, help="Measured adapter repetitions; final mask comes from the last")
    parser.add_argument("--confirm-run", action="store_true", help="Required for more than five model-image executions")
    parser.add_argument("--sealed-authority", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def _resolve_images(args: argparse.Namespace) -> list[dict[str, Any]]:
    if bool(args.image) == bool(args.manifest):
        raise ValueError("Use exactly one input source: --manifest or one/more --image")
    if args.image:
        if args.image_ids:
            raise ValueError("--image-ids only applies to a manifest")
        items = []
        for source in args.image:
            path = source if source.is_absolute() else REPO_ROOT / source
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Image is missing: {path}")
            items.append({"image_id": path.stem, "image_path": path, "mask_paths": []})
    else:
        if args.data_root is None:
            raise ValueError("--data-root is required with --manifest")
        payload = load_json(args.manifest)
        if payload.get("dataset_id") != args.dataset or payload.get("split") != args.split:
            raise ValueError("manifest dataset/split does not match the requested run")
        validate_dataset_manifest(payload, data_root=args.data_root, require_files=True)
        requested = set(args.image_ids.split(",")) if args.image_ids else None
        items = []
        for item in payload["items"]:
            if requested is not None and item["image_id"] not in requested:
                continue
            items.append({
                **item,
                "image_path": (args.data_root / item["image_path"]).resolve(),
                "mask_paths": [(args.data_root / path).resolve() for path in item.get("mask_paths", [])],
            })
        if requested is not None:
            missing = requested - {item["image_id"] for item in items}
            if missing:
                raise ValueError("unknown image ids: " + ", ".join(sorted(missing)))
    if len({item["image_id"] for item in items}) != len(items):
        raise ValueError("input image ids are not unique")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        items = items[:args.limit]
    if not items:
        raise ValueError("no images selected")
    return items


def _select_methods(selector: str) -> list[dict[str, Any]]:
    methods = load_benchmark_methods(REPO_ROOT / "configs" / "segmentation_models.json")
    if selector == "all":
        return methods
    aliases = {alias: method for method in methods for alias in (method["method_id"], method["backend_id"], method["id"])}
    selected = []
    for value in selector.split(","):
        if value not in aliases:
            raise ValueError(f"unknown model: {value}")
        if aliases[value] not in selected:
            selected.append(aliases[value])
    return selected


def _checkpoint(method: dict[str, Any]) -> dict[str, Any]:
    command = method.get("adapter_command", [])
    if "--checkpoint" not in command:
        return {"path": None, "sha256": None, "available": method["method_id"] == "S16"}
    token = command[command.index("--checkpoint") + 1].replace("{repo_root}", str(REPO_ROOT))
    path = Path(token)
    return {
        "path": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
        "sha256": sha256_file(path) if path.is_file() else None,
        "available": path.is_file(),
    }


def _b2_methods(methods: list[dict[str, Any]], path: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if path is None:
        raise ValueError("No disponible: faltan checkpoints B2; proporcione --b2-config con checkpoints realmente reentrenados")
    config = load_json(path)
    if config.get("evaluation") != "B2" or config.get("frozen") is not True:
        raise ValueError("--b2-config debe declarar evaluation=B2 y frozen=true")
    identities = config.get("b2_checkpoints", {})
    expected = {method["method_id"] for method in methods if method["kind"] == "neural"}
    if set(identities) != expected:
        raise ValueError(f"Checkpoints B2 incompletos; faltan {sorted(expected - set(identities))}")
    replaced = []
    for method in methods:
        if method["kind"] != "neural":
            replaced.append(method); continue
        identity = identities[method["method_id"]]
        members = identity.get("members") or [identity]
        if len(members) not in {1, 5}:
            raise ValueError(f"B2 requiere un checkpoint final o cinco folds para {method['method_id']}")
        commands = []
        member_folds = set()
        for member in members:
            checkpoint = Path(member["path"])
            if not checkpoint.is_absolute(): checkpoint = REPO_ROOT / checkpoint
            if not checkpoint.is_file() or sha256_file(checkpoint) != member.get("sha256"):
                raise ValueError(f"Checkpoint B2 ausente o modificado: {method['method_id']}")
            metadata = Path(member.get("metadata_path", checkpoint.with_suffix(checkpoint.suffix + ".metadata.json")))
            if not metadata.is_absolute(): metadata = REPO_ROOT / metadata
            if not metadata.is_file() or (member.get("metadata_sha256") and sha256_file(metadata) != member["metadata_sha256"]):
                raise ValueError(f"Falta identidad de entrenamiento B2 o su hash cambió: {metadata}")
            metadata_payload = load_json(metadata)
            if metadata_payload.get("protocol") != "B2" or metadata_payload.get("method_id") != method["method_id"] or metadata_payload.get("checkpoint_sha256") != member["sha256"]:
                raise ValueError(f"Identidad de entrenamiento B2 incompatible: {metadata}")
            member_folds.add(metadata_payload.get("fold"))
            command = list(method["adapter_command"])
            command[command.index("--checkpoint") + 1] = str(checkpoint)
            commands.append(command)
        if len(members) == 5 and member_folds != set(range(5)):
            raise ValueError(f"B2 requiere los cinco folds 0-4 para {method['method_id']}")
        clone = {**method, "adapter_command": commands[0], "ensemble_commands": commands if len(commands) == 5 else None, "checkpoint_status": "b2_verified"}
        replaced.append(clone)
    return replaced, config


def _versions() -> dict[str, str | None]:
    versions = {"python": sys.version.split()[0], "opencv": cv2.__version__, "numpy": np.__version__}
    for package in ("pillow", "torch", "transformers", "segmentation-models-pytorch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout
    source_files = [path for root in ("configs", "scripts", "src", "web") for path in (REPO_ROOT / root).rglob("*") if path.is_file() and "__pycache__" not in path.parts]
    working_tree_hash = content_hash({path.relative_to(REPO_ROOT).as_posix(): sha256_file(path) for path in sorted(source_files)})
    return {"commit": commit, "dirty": bool(status), "status_lines": status.splitlines(), "working_tree_source_hash": working_tree_hash}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        atomic_write_bytes(path, b"")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_bytes(path, stream.getvalue().encode("utf-8"))


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"invalid mask: {path}")
    if mask.shape != (size[1], size[0]):
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)


def _write_rgb_preview(path: Path, rgb: np.ndarray) -> None:
    preview = rgb
    if max(rgb.shape[:2]) > 1024:
        scale = 1024 / max(rgb.shape[:2])
        preview = cv2.resize(rgb, (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    success, encoded = cv2.imencode(".jpg", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success: raise RuntimeError(f"No se pudo codificar preview web: {path}")
    atomic_write_bytes(path, encoded.tobytes())


def _prediction_overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = rgb.copy()
    foreground = mask > 0
    overlay[foreground] = np.clip(0.45 * overlay[foreground] + 0.55 * np.array([255, 40, 40]), 0, 255).astype(np.uint8)
    return overlay


def _neural_backend(method: dict[str, Any], input_path: Path, output_path: Path, *, warmup: int = 0, repetitions: int = 1) -> BackendResult:
    if method.get("ensemble_commands"):
        members = []
        for index, command in enumerate(method["ensemble_commands"]):
            member_path = output_path.with_name(f"ensemble_member_{index}.png")
            member = _neural_backend({**method, "adapter_command": command, "ensemble_commands": None}, input_path, member_path, warmup=warmup, repetitions=repetitions)
            if member.failure_code: return member
            members.append(member)
        masks = [cv2.resize(member.native_mask, members[0].input_size, interpolation=cv2.INTER_NEAREST) for member in members]
        final = (np.mean(masks, axis=0) >= 0.5).astype(np.uint8)
        atomic_write_png(output_path, final * 255)
        return BackendResult(native_mask=final, input_size=members[0].input_size, backend_time_ms=sum(member.backend_time_ms for member in members), model_identity={"method_id": method["method_id"], "backend": method["backend_id"], "protocol": "B2", "ensemble_members": len(members)}, details={"ensemble": "majority vote of five grouped-fold B2 checkpoints", "members": [member.manifest() for member in members]})
    if warmup < 0 or repetitions < 1: raise ValueError("warmup/repetitions inválidos")
    samples = []
    runtime = None
    for _ in range(repetitions):
        success, message, runtime = review.run_adapter(method, input_path, output_path, internal_warmup=warmup)
        if not success: break
        samples.append(runtime)
    elapsed_samples = [float(item["wall_seconds"] * 1000) for item in samples]
    elapsed = float(np.median(elapsed_samples)) if elapsed_samples else float(runtime["wall_seconds"] * 1000) if runtime else 0.0
    if not success:
        return BackendResult(
            native_mask=np.zeros((1, 1), np.uint8), input_size=(1, 1), backend_time_ms=elapsed,
            model_identity={"method_id": method["method_id"], "backend": method["backend_id"]},
            warnings=[message], failure_code="adapter_error", details={"runtime": runtime},
        )
    mask = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("adapter output could not be decoded")
    return BackendResult(
        native_mask=(mask > 127).astype(np.uint8), input_size=(mask.shape[1], mask.shape[0]), backend_time_ms=elapsed, threshold=None if method["backend_id"].startswith("theodore-") else 0.5,
        model_identity={"method_id": method["method_id"], "backend": method["backend_id"]},
        details={
            "runtime": runtime, "native_processing": "adapter contract preserved",
            "native_decision_rule": "two-class argmax" if method["backend_id"].startswith("theodore-") else "foreground probability >= 0.5",
            "warmup_repetitions": warmup, "measurement_repetitions": repetitions,
            "backend_time_samples_ms": elapsed_samples,
            "backend_time_p25_ms": float(np.percentile(elapsed_samples, 25)),
            "backend_time_p75_ms": float(np.percentile(elapsed_samples, 75)),
            "backend_time_p95_ms": float(np.percentile(elapsed_samples, 95)),
            "peak_ram_mb": max(item["peak_ram_mb"] for item in samples),
            "vram_peak_mb": max((item.get("adapter", {}).get("peak_vram_mb") or 0) for item in samples) or None,
            "vram_status": "measured_by_adapter" if any(item.get("adapter", {}).get("peak_vram_mb") is not None for item in samples) else "not_available_cpu_or_legacy_adapter",
            "cpu_measurement": "child process CPU time and sampled process-tree RSS",
            "model_load_time_samples_ms": [item.get("adapter", {}).get("model_load_time_ms") for item in samples],
            "inference_time_samples_ms": [item.get("adapter", {}).get("inference_time_ms") for item in samples],
            "timing_scope": "backend_time is adapter subprocess wall time including startup/load; adapter sidecar separately records synchronized model load and forward; end_to_end is reported separately",
        },
    )


def main() -> None:
    args = parse_args()
    datasets = validate_dataset_registry(load_json(REPO_ROOT / "configs" / "benchmark" / "datasets.json"))
    dataset = next((item for item in datasets if item["id"] == args.dataset), None)
    if dataset is None:
        raise SystemExit(f"Dataset no registrado: {args.dataset}")
    if not dataset["enabled"]:
        raise SystemExit(f"Dataset deshabilitado: {dataset['warning']}")
    if args.split == "test":
        if args.evaluation != "B2" or args.sealed_authority is None:
            raise SystemExit("El test solo puede ejecutarse mediante sealed_test.py después de congelar B2.")
        authority = load_json(args.sealed_authority)
        if authority.get("status") != "executing" or authority.get("execution_locked"):
            raise SystemExit("La autoridad sellada no está en estado executing.")
    try:
        images = _resolve_images(args)
        methods = _select_methods(args.models)
        b2_config = None
        if args.evaluation == "B2":
            all_methods, b2_config = _b2_methods(load_benchmark_methods(REPO_ROOT / "configs" / "segmentation_models.json"), args.b2_config)
            selected_ids = {method["method_id"] for method in methods}
            methods = [method for method in all_methods if method["method_id"] in selected_ids]
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    config = load_json(args.config)
    if b2_config and sha256_file(args.config) != b2_config.get("base_config_sha256"):
        raise SystemExit("La configuración de evaluación no coincide con la configuración base congelada B2.")
    detector = detector_from_config(config, REPO_ROOT) if args.evaluation in {"B1", "B2"} else None
    resources = {method["method_id"]: _checkpoint(method) for method in methods}
    summary = {
        "evaluation": args.evaluation,
        "dataset": args.dataset,
        "split": args.split,
        "images": [item["image_id"] for item in images],
        "methods": [item["method_id"] for item in methods],
        "executions": len(images) * len(methods),
        "resources": resources,
        "configuration_hash": content_hash(config),
        "dataset_manifest_hash": content_hash(load_json(args.manifest)) if args.manifest else None,
        "b2_configuration_hash": content_hash(b2_config) if b2_config else None,
    }
    if args.dry_run:
        print(json.dumps({**summary, "dry_run": True}, indent=2, ensure_ascii=False))
        return
    if summary["executions"] > 5 and not args.confirm_run:
        raise SystemExit(
            f"La corrida iniciaría {summary['executions']} ejecuciones. Revise primero --dry-run y repita con --confirm-run."
        )

    artifact_root = args.artifact_root or Path(config["artifact_root"])
    if not artifact_root.is_absolute():
        artifact_root = REPO_ROOT / artifact_root
    run_id = args.run_id or f"{args.evaluation.lower()}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise SystemExit("--run-id contains unsafe characters")
    run_directory = artifact_root / "runs" / run_id
    if run_directory.exists():
        raise SystemExit(f"Run already exists and will not be overwritten: {run_directory}")
    run_directory.mkdir(parents=True)
    manifest_path = run_directory / "run_manifest.json"
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "created",
        **summary,
        "configuration": config,
        "git": _git_state(),
        "versions": _versions(),
        "device": "cpu",
        "seed": config["seed"],
        "image_list_hash": content_hash(summary["images"]),
        "failures": [],
        "fallbacks": [],
    }
    atomic_write_json(manifest_path, manifest)
    manifest["status"] = "running"
    atomic_write_json(manifest_path, manifest)
    metrics_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    completed = 0
    started_run = time.perf_counter()
    try:
        for item in images:
            rgb = _load_rgb(item["image_path"])
            if config.get("store_web_previews", True):
                _write_rgb_preview(run_directory / "inputs" / f"{item['image_id']}.jpg", rgb)
            ground_truth = _load_mask(item["mask_paths"][0], (rgb.shape[1], rgb.shape[0])) if item["mask_paths"] else None
            if args.split == "test":
                ground_truth = None
            if ground_truth is not None and config.get("store_web_previews", True):
                atomic_write_png(run_directory / "inputs" / f"{item['image_id']}.ground_truth.png", ground_truth * 255)
            image = ImageInput(
                item["image_id"], args.dataset, args.split, rgb,
                ground_truth=ground_truth, ground_truth_allowed=ground_truth is not None,
                metadata={key: value for key, value in item.items() if key not in {"image_path", "mask_paths"}},
            )
            p0 = run_p0(
                image, config, detector, cache_root=artifact_root / "preprocessing", reuse_cache=args.reuse_p0_cache
            ) if args.evaluation in {"B1", "B2"} else None
            for method in methods:
                prediction_directory = run_directory / "predictions" / method["method_id"] / image.image_id
                prediction_directory.mkdir(parents=True)
                native_path = prediction_directory / "native_mask.png"
                method_started = time.perf_counter()
                if method["method_id"] == "S16":
                    backend = grabcut_classic(rgb, **{
                        "margin_fraction": config["grabcut"]["classic"]["margin_fraction"],
                        "iterations": config["grabcut"]["classic"]["iterations"],
                        "nearly_complete_fraction": config["grabcut"]["nearly_complete_fraction"],
                    }) if args.evaluation == "A" else grabcut_robust(
                        p0,
                        iterations=config["grabcut"]["robust"]["iterations"],
                        color_space=config["grabcut"]["robust"]["color_space"],
                        nearly_complete_fraction=config["grabcut"]["nearly_complete_fraction"],
                    )
                    atomic_write_png(native_path, backend.native_mask * 255)
                else:
                    adapter_input = item["image_path"] if args.evaluation == "A" else p0.cache_directory / "roi_input.png"
                    backend = _neural_backend(method, adapter_input, native_path, warmup=args.warmup, repetitions=args.repetitions)
                post_started = time.perf_counter()
                if backend.failure_code:
                    restored = np.zeros(rgb.shape[:2], np.uint8)
                    final = restored
                    post_details = {"skipped": True, "reason": backend.failure_code}
                elif args.evaluation == "A":
                    final = cv2.resize(backend.native_mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                    post_details = {"applied": False, "reason": "Evaluation A preserves native behavior"}
                else:
                    restored = p0.transform.roi_to_original_mask(backend.native_mask)
                    atomic_write_png(prediction_directory / "pre_postprocess_mask.png", restored * 255)
                    final, post_details = common_postprocess(
                        restored, p0.fov_mask, config["postprocessing"], selected_bbox=p0.selected_bbox_original
                    )
                post_time = (time.perf_counter() - post_started) * 1000
                atomic_write_png(prediction_directory / "final_mask.png", final * 255)
                if config.get("store_web_previews", True):
                    _write_rgb_preview(prediction_directory / "overlay.jpg", _prediction_overlay(rgb, final))
                result_metrics = segmentation_metrics(
                    final, ground_truth,
                    fov_mask=p0.fov_mask if p0 else None,
                    hair_mask=p0.hair_mask if p0 else None,
                    threshold_jaccard_cutoff=config["metrics"]["threshold_jaccard_cutoff"],
                    boundary_tolerance_diagonal_fraction=config["metrics"]["boundary_tolerance_diagonal_fraction"],
                ) if ground_truth is not None else None
                end_to_end = (time.perf_counter() - method_started) * 1000 + (sum(p0.timings_ms.values()) if p0 else 0.0)
                result_payload = {
                    "schema_version": 1,
                    "evaluation": args.evaluation,
                    "dataset_id": args.dataset,
                    "split": args.split,
                    "image_id": image.image_id,
                    "method_id": method["method_id"],
                    "backend": backend.manifest(),
                    "postprocessing": post_details,
                    "postprocessing_time_ms": post_time,
                    "end_to_end_time_ms": end_to_end,
                    "metrics": result_metrics,
                    "p0_cache_key": p0.cache_key if p0 else None,
                    "p0_cache_hit": p0.cache_hit if p0 else None,
                    "p0_fallback_used": p0.fallback_used if p0 else False,
                    "metadata": image.metadata,
                    "original_preview": f"inputs/{image.image_id}.jpg" if config.get("store_web_previews", True) else None,
                    "ground_truth_preview": f"inputs/{image.image_id}.ground_truth.png" if ground_truth is not None and config.get("store_web_previews", True) else None,
                    "overlay_preview": "overlay.jpg" if config.get("store_web_previews", True) else None,
                    "timing_note": (
                        "P0 cache reused; end-to-end includes the recorded uncached P0 stage times."
                        if p0 and p0.cache_hit else "Measured sequentially without P0 cache reuse."
                    ),
                    "failure_code": backend.failure_code,
                }
                atomic_write_json(prediction_directory / "result.json", result_payload)
                if result_metrics:
                    metrics_rows.append({
                        "image_id": image.image_id,
                        "method_id": method["method_id"],
                        **{key: value for key, value in result_metrics.items() if isinstance(value, (int, float)) or value is None},
                    })
                if backend.failure_code:
                    failure = {"image_id": image.image_id, "method_id": method["method_id"], "failure_code": backend.failure_code, "warnings": " | ".join(backend.warnings)}
                    failures.append(failure)
                if p0 and p0.fallback_used:
                    manifest["fallbacks"].append({"image_id": image.image_id, "method_id": method["method_id"], "type": "yolo_full_fov"})
                completed += 1
                elapsed = time.perf_counter() - started_run
                eta = elapsed / completed * (summary["executions"] - completed)
                print(f"[{completed}/{summary['executions']}] {method['method_id']} {image.image_id} · ETA {eta:.1f}s", flush=True)
        _write_csv(run_directory / "metrics_per_image.csv", metrics_rows)
        _write_csv(run_directory / "failures.csv", failures)
        manifest["status"] = "completed"
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["failures"] = failures
        write_report(run_directory, repetitions=config["metrics"]["bootstrap_repetitions"], seed=config["seed"])
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["fatal_error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        atomic_write_json(manifest_path, manifest)
    print(json.dumps({"run_id": run_id, "status": manifest["status"], "directory": str(run_directory), "failures": len(failures)}, indent=2))


if __name__ == "__main__":
    main()
