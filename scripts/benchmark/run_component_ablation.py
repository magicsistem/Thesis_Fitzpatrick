#!/usr/bin/env python3
"""Run the five pre-registered one-component-out B1 ablations.

Conditions:
  B1_FULL, B1_NO_HAIR, B1_NO_YOLO, B1_NO_FOV, B1_NO_POST.

This runner refuses historical per-fold YOLO configurations.  The input config
must have been generated from a frozen ``model_role=final_refit`` checkpoint.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "benchmark"))

from thesis_fitzpatrick.benchmark import (  # noqa: E402
    ImageInput,
    atomic_write_bytes,
    atomic_write_json,
    content_hash,
    is_fatal_backend_failure,
    load_json,
    validate_dataset_registry,
)
from thesis_fitzpatrick.grabcut import common_postprocess, grabcut_robust  # noqa: E402
from thesis_fitzpatrick.masks import build_clean_skin_mask, measure_skin_colour, prepare_clean_skin_context  # noqa: E402
from thesis_fitzpatrick.metrics import segmentation_metrics  # noqa: E402
from thesis_fitzpatrick.post05 import COMPONENT_ABLATIONS, require_final_b1_config  # noqa: E402
from thesis_fitzpatrick.preprocessing import atomic_write_png, detector_from_config, run_p0  # noqa: E402
from thesis_fitzpatrick.reporting import write_report  # noqa: E402
import run_evaluation as evaluation_cli  # noqa: E402


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", default=",".join(COMPONENT_ABLATIONS))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--models", default="all")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--image", action="append", type=Path, default=[])
    parser.add_argument("--image-ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--reuse-p0-cache", action="store_true")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    registry = validate_dataset_registry(load_json(REPO_ROOT / "configs" / "benchmark" / "datasets.json"))
    dataset = next((item for item in registry if item["id"] == args.dataset), None)
    if dataset is None or not dataset["enabled"]:
        raise SystemExit(f"Dataset not registered/enabled: {args.dataset}")
    if args.split == "test":
        raise SystemExit("Component ablations are development-only and refuse the sealed test")
    conditions = [value.strip() for value in args.conditions.split(",") if value.strip()]
    if not conditions or any(value not in COMPONENT_ABLATIONS for value in conditions):
        raise SystemExit("--conditions must be a comma-separated subset of " + ",".join(COMPONENT_ABLATIONS))
    try:
        items = evaluation_cli._resolve_images(args)
        methods = evaluation_cli._select_methods(args.models)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    config = load_json(args.config)
    try:
        require_final_b1_config(config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    detector = detector_from_config(config, REPO_ROOT)
    resources = {method["method_id"]: evaluation_cli._checkpoint(method) for method in methods}
    summary = {
        "evaluation": "post05_component_ablation",
        "conditions": conditions,
        "dataset": args.dataset,
        "split": args.split,
        "methods": [method["method_id"] for method in methods],
        "images": [item["image_id"] for item in items],
        "executions": len(conditions) * len(items) * len(methods),
        "configuration_hash": content_hash(config),
        "dataset_manifest_hash": content_hash(load_json(args.manifest)) if args.manifest else None,
        "resources": resources,
        "ablation_contract": COMPONENT_ABLATIONS,
        "yolo_final_identity_hash": config["post05_contract"]["yolo_identity_hash"],
    }
    if args.dry_run:
        print(json.dumps({**summary, "dry_run": True}, indent=2, ensure_ascii=False))
        return
    if summary["executions"] > 5 and not args.confirm_run:
        raise SystemExit(f"Run would start {summary['executions']} executions; review --dry-run and repeat with --confirm-run")

    artifact_root = args.artifact_root or Path(config["artifact_root"])
    if not artifact_root.is_absolute():
        artifact_root = REPO_ROOT / artifact_root
    run_id = args.run_id or f"post05-ablation-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise SystemExit("--run-id contains unsafe characters")
    run_directory = artifact_root / "runs" / run_id
    if run_directory.exists():
        raise SystemExit(f"Run already exists and will not be overwritten: {run_directory}")
    run_directory.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        **summary,
        "configuration": config,
        "git": evaluation_cli._git_state(),
        "versions": evaluation_cli._versions(),
        "failures": [],
        "quality_flags": [],
        "fallbacks": [],
    }
    atomic_write_json(run_directory / "run_manifest.json", manifest)
    metrics_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    quality_flags: list[dict[str, Any]] = []
    completed = 0
    started = time.perf_counter()

    try:
        for item in items:
            rgb = evaluation_cli._load_rgb(item["image_path"])
            if config.get("store_web_previews", True):
                evaluation_cli._write_rgb_preview(run_directory / "inputs" / f"{item['image_id']}.jpg", rgb)
            gt = evaluation_cli._load_mask(item["mask_paths"][0], (rgb.shape[1], rgb.shape[0])) if item["mask_paths"] else None
            if gt is not None and config.get("store_web_previews", True):
                atomic_write_png(run_directory / "inputs" / f"{item['image_id']}.ground_truth.png", gt * 255)
            image = ImageInput(
                item["image_id"], args.dataset, args.split, rgb,
                ground_truth=gt, ground_truth_allowed=gt is not None,
                metadata={key: value for key, value in item.items() if key not in {"image_path", "mask_paths"}},
            )
            clean_skin_context = prepare_clean_skin_context(Image.fromarray(rgb, mode="RGB"))

            for condition in conditions:
                spec = COMPONENT_ABLATIONS[condition]
                p0 = run_p0(
                    image,
                    config,
                    detector,
                    cache_root=artifact_root / "preprocessing",
                    reuse_cache=args.reuse_p0_cache,
                    enable_fov=spec["enable_fov"],
                    enable_hair=spec["enable_hair"],
                    enable_yolo=spec["enable_yolo"],
                )
                for method in methods:
                    output = run_directory / "predictions" / condition / method["method_id"] / image.image_id
                    output.mkdir(parents=True)
                    native_path = output / "native_mask.png"
                    method_started = time.perf_counter()
                    if method["method_id"] == "S16":
                        backend = grabcut_robust(
                            p0,
                            iterations=config["grabcut"]["robust"]["iterations"],
                            color_space=config["grabcut"]["robust"]["color_space"],
                            nearly_complete_fraction=config["grabcut"]["nearly_complete_fraction"],
                        )
                        atomic_write_png(native_path, backend.native_mask * 255)
                    else:
                        adapter_input = p0.cache_directory / "roi_input.png"
                        backend = evaluation_cli._neural_backend(
                            method, adapter_input, native_path, warmup=args.warmup, repetitions=args.repetitions
                        )
                    fatal = is_fatal_backend_failure(backend.failure_code)
                    if fatal:
                        restored = np.zeros(rgb.shape[:2], dtype=np.uint8)
                        final = restored
                        post = {"skipped": True, "reason": backend.failure_code}
                    else:
                        restored = p0.transform.roi_to_original_mask(backend.native_mask)
                        atomic_write_png(output / "pre_postprocess_mask.png", restored * 255)
                        if spec["apply_post"]:
                            final, post = common_postprocess(
                                restored, p0.fov_mask, config["postprocessing"], selected_bbox=p0.selected_bbox_original
                            )
                        else:
                            final = restored
                            post = {"applied": False, "reason": "B1_NO_POST pre-registered ablation"}
                    atomic_write_png(output / "final_mask.png", final * 255)

                    clean_skin, clean_meta = build_clean_skin_mask(Image.fromarray(rgb, mode="RGB"), final, context=clean_skin_context)
                    # The clean-skin artifact follows the stages that are enabled
                    # in each ablation; it is descriptive and is not part of the
                    # TOP-3 segmentation score.
                    if spec["enable_fov"]:
                        clean_skin &= p0.fov_mask.astype(bool)
                    if spec["enable_hair"]:
                        clean_skin &= ~p0.hair_mask.astype(bool)
                    clean_skin &= ~final.astype(bool)
                    atomic_write_png(output / "clean_skin_mask.png", clean_skin * 255)
                    skin_colour = None
                    skin_colour_error = None
                    try:
                        skin_colour = measure_skin_colour(Image.fromarray(rgb, mode="RGB"), clean_skin).to_dict()
                    except ValueError as exc:
                        skin_colour_error = str(exc)
                    if config.get("store_web_previews", True):
                        evaluation_cli._write_rgb_preview(output / "overlay.jpg", evaluation_cli._prediction_overlay(rgb, final))
                    metrics = segmentation_metrics(
                        final,
                        gt,
                        fov_mask=p0.fov_mask if spec["enable_fov"] else None,
                        hair_mask=p0.hair_mask if spec["enable_hair"] else None,
                        threshold_jaccard_cutoff=config["metrics"]["threshold_jaccard_cutoff"],
                        boundary_tolerance_diagonal_fraction=config["metrics"]["boundary_tolerance_diagonal_fraction"],
                    ) if gt is not None and not fatal else None
                    flags = (metrics or {}).get("flags") or {}
                    outcome = (
                        "technical_failure" if fatal
                        else "degenerate_prediction" if backend.failure_code or flags.get("prediction_empty") or flags.get("prediction_nearly_complete")
                        else "ok"
                    )
                    result = {
                        "schema_version": 1,
                        "evaluation": "post05_component_ablation",
                        "condition": condition,
                        "dataset_id": args.dataset,
                        "split": args.split,
                        "method_id": method["method_id"],
                        "image_id": image.image_id,
                        "metadata": image.metadata,
                        "backend": backend.manifest(),
                        "p0_stages": p0.stages,
                        "p0_stage_details": p0.stage_details,
                        "p0_cache_key": p0.cache_key,
                        "p0_cache_hit": p0.cache_hit,
                        "p0_fallback_used": p0.fallback_used,
                        "postprocessing": post,
                        "metrics": metrics,
                        "failure_code": backend.failure_code,
                        "failure_is_fatal": fatal,
                        "outcome": outcome,
                        "end_to_end_time_ms": (time.perf_counter() - method_started) * 1000 + sum(p0.timings_ms.values()),
                        "clean_skin_mask": "clean_skin_mask.png",
                        "clean_skin_metadata": clean_meta,
                        "skin_colour": skin_colour,
                        "skin_colour_available": skin_colour is not None,
                        "skin_colour_error": skin_colour_error,
                        "overlay_preview": "overlay.jpg" if config.get("store_web_previews", True) else None,
                    }
                    atomic_write_json(output / "result.json", result)
                    if metrics:
                        metrics_rows.append({
                            "condition": condition,
                            "image_id": image.image_id,
                            "method_id": method["method_id"],
                            **{key: value for key, value in metrics.items() if isinstance(value, (int, float)) or value is None},
                        })
                    if fatal:
                        failures.append({
                            "condition": condition,
                            "image_id": image.image_id,
                            "method_id": method["method_id"],
                            "failure_code": backend.failure_code,
                            "warnings": " | ".join(backend.warnings),
                        })
                    elif outcome == "degenerate_prediction":
                        reasons = [value for value in (
                            backend.failure_code,
                            "prediction_empty" if flags.get("prediction_empty") else None,
                            "prediction_nearly_complete" if flags.get("prediction_nearly_complete") else None,
                        ) if value]
                        quality_flags.append({
                            "condition": condition,
                            "image_id": image.image_id,
                            "method_id": method["method_id"],
                            "quality_flags": " | ".join(dict.fromkeys(reasons)),
                            "warnings": " | ".join(backend.warnings),
                        })
                    if p0.fallback_used:
                        manifest["fallbacks"].append({
                            "condition": condition,
                            "image_id": image.image_id,
                            "method_id": method["method_id"],
                            "type": "yolo_full_fov" if spec["enable_yolo"] else "not_applicable_yolo_disabled",
                        })
                    completed += 1
                    eta = (time.perf_counter() - started) / completed * (summary["executions"] - completed)
                    print(f"[{completed}/{summary['executions']}] {condition} {method['method_id']} {image.image_id} · ETA {eta:.1f}s", flush=True)

        _write_csv(run_directory / "metrics_per_image.csv", metrics_rows)
        _write_csv(run_directory / "failures.csv", failures)
        _write_csv(run_directory / "quality_flags.csv", quality_flags)
        manifest["failures"] = failures
        manifest["quality_flags"] = quality_flags
        if failures:
            raise RuntimeError(f"Component ablation incomplete: {len(failures)} technical failures require rerun")
        manifest["status"] = "completed"
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_report(run_directory, repetitions=config["metrics"]["bootstrap_repetitions"], seed=config["seed"])
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["fatal_error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        atomic_write_json(run_directory / "run_manifest.json", manifest)
    print(json.dumps({"run_id": run_id, "status": manifest["status"], "directory": str(run_directory), "failures": len(failures)}, indent=2))


if __name__ == "__main__":
    main()
