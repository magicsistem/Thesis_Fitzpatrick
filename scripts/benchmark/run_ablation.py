#!/usr/bin/env python3
"""Run C0-C3 for any or all S01-S16 backends."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from thesis_fitzpatrick.benchmark import ImageInput, atomic_write_json, content_hash, load_json, validate_dataset_registry  # noqa: E402
from thesis_fitzpatrick.grabcut import common_postprocess, grabcut_classic, grabcut_robust  # noqa: E402
from thesis_fitzpatrick.metrics import segmentation_metrics  # noqa: E402
from thesis_fitzpatrick.preprocessing import atomic_write_png, detector_from_config, run_p0  # noqa: E402
from thesis_fitzpatrick.reporting import write_report  # noqa: E402
import run_evaluation as evaluation_cli  # noqa: E402


STAGES = {
    "C1": {"enable_fov": True, "enable_hair": False, "enable_yolo": False},
    "C2": {"enable_fov": True, "enable_hair": True, "enable_yolo": False},
    "C3": {"enable_fov": True, "enable_hair": True, "enable_yolo": True},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", default="C0,C1,C2,C3")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--models", default="all", help="all, S01,...,S16, or catalogue backend ids")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--image", action="append", type=Path, default=[])
    parser.add_argument("--image-ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "benchmark" / "default.json")
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--reuse-p0-cache", action="store_true")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = validate_dataset_registry(load_json(REPO_ROOT / "configs" / "benchmark" / "datasets.json"))
    dataset = next((item for item in datasets if item["id"] == args.dataset), None)
    if dataset is None or not dataset["enabled"]:
        raise SystemExit(f"Dataset no registrado o deshabilitado: {args.dataset}")
    if args.split == "test":
        raise SystemExit("Las ablaciones no pueden ejecutarse sobre el test sellado.")
    conditions = args.conditions.split(",")
    if not conditions or any(condition not in {"C0", "C1", "C2", "C3"} for condition in conditions):
        raise SystemExit("--conditions must be a comma-separated subset of C0,C1,C2,C3")
    try:
        items = evaluation_cli._resolve_images(args)
        methods = evaluation_cli._select_methods(args.models)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    config = load_json(args.config)
    detector = detector_from_config(config, REPO_ROOT) if "C3" in conditions else None
    summary = {
        "evaluation": "ablation",
        "conditions": conditions,
        "dataset": args.dataset,
        "split": args.split,
        "methods": [method["method_id"] for method in methods],
        "images": [item["image_id"] for item in items],
        "executions": len(conditions) * len(items) * len(methods),
        "configuration_hash": content_hash(config),
        "dataset_manifest_hash": content_hash(load_json(args.manifest)) if args.manifest else None,
    }
    if args.dry_run:
        print(json.dumps({**summary, "dry_run": True}, indent=2, ensure_ascii=False))
        return
    if summary["executions"] > 5 and not args.confirm_run:
        raise SystemExit(f"La corrida iniciaría {summary['executions']} ejecuciones; revise --dry-run y repita con --confirm-run.")
    artifact_root = args.artifact_root or Path(config["artifact_root"])
    if not artifact_root.is_absolute():
        artifact_root = REPO_ROOT / artifact_root
    run_id = args.run_id or f"ablation-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
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
        "failures": [],
    }
    atomic_write_json(run_directory / "run_manifest.json", manifest)
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
            image = ImageInput(item["image_id"], args.dataset, args.split, rgb, ground_truth=gt, ground_truth_allowed=gt is not None, metadata={key: value for key, value in item.items() if key not in {"image_path", "mask_paths"}})
            for condition in conditions:
                if condition == "C0":
                    p0 = None
                else:
                    p0 = run_p0(image, config, detector, cache_root=artifact_root / "preprocessing", reuse_cache=args.reuse_p0_cache, **STAGES[condition])
                for method in methods:
                    output = run_directory / "predictions" / condition / method["method_id"] / image.image_id
                    output.mkdir(parents=True)
                    started_method = time.perf_counter()
                    if method["method_id"] == "S16":
                        backend = grabcut_classic(
                            rgb,
                            margin_fraction=config["grabcut"]["classic"]["margin_fraction"],
                            iterations=config["grabcut"]["classic"]["iterations"],
                            nearly_complete_fraction=config["grabcut"]["nearly_complete_fraction"],
                        ) if condition == "C0" else grabcut_robust(
                            p0,
                            iterations=config["grabcut"]["robust"]["iterations"],
                            color_space=config["grabcut"]["robust"]["color_space"],
                            nearly_complete_fraction=config["grabcut"]["nearly_complete_fraction"],
                        )
                        atomic_write_png(output / "native_mask.png", backend.native_mask * 255)
                    else:
                        adapter_input = item["image_path"] if condition == "C0" else p0.cache_directory / "roi_input.png"
                        backend = evaluation_cli._neural_backend(
                            method, adapter_input, output / "native_mask.png",
                            warmup=args.warmup, repetitions=args.repetitions,
                        )
                    if backend.failure_code:
                        final = np.zeros(rgb.shape[:2], dtype=np.uint8)
                        post = {"skipped": True, "reason": backend.failure_code}
                    elif condition == "C0":
                        final = cv2.resize(backend.native_mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                        post = {"applied": False, "reason": "C0 preserves native backend output"}
                    else:
                        restored = p0.transform.roi_to_original_mask(backend.native_mask)
                        atomic_write_png(output / "pre_postprocess_mask.png", restored * 255)
                        final, post = common_postprocess(restored, p0.fov_mask, config["postprocessing"], selected_bbox=p0.selected_bbox_original)
                    atomic_write_png(output / "final_mask.png", final * 255)
                    if config.get("store_web_previews", True):
                        evaluation_cli._write_rgb_preview(output / "overlay.jpg", evaluation_cli._prediction_overlay(rgb, final))
                    metrics = segmentation_metrics(
                        final, gt,
                        fov_mask=p0.fov_mask if p0 else None,
                        hair_mask=p0.hair_mask if p0 else None,
                        threshold_jaccard_cutoff=config["metrics"]["threshold_jaccard_cutoff"],
                        boundary_tolerance_diagonal_fraction=config["metrics"]["boundary_tolerance_diagonal_fraction"],
                    ) if gt is not None else None
                    atomic_write_json(output / "result.json", {
                        "schema_version": 1, "condition": condition, "dataset_id": args.dataset, "split": args.split, "method_id": method["method_id"], "image_id": image.image_id,
                        "original_preview": f"inputs/{image.image_id}.jpg" if config.get("store_web_previews", True) else None,
                        "ground_truth_preview": f"inputs/{image.image_id}.ground_truth.png" if gt is not None and config.get("store_web_previews", True) else None,
                        "overlay_preview": "overlay.jpg" if config.get("store_web_previews", True) else None,
                        "metadata": image.metadata, "backend": backend.manifest(),
                        "p0_stages": p0.stages if p0 else {"fov": False, "hair": False, "yolo": False},
                        "p0_cache_key": p0.cache_key if p0 else None, "p0_fallback_used": p0.fallback_used if p0 else False, "postprocessing": post, "metrics": metrics,
                        "end_to_end_time_ms": (time.perf_counter() - started_method) * 1000 + (sum(p0.timings_ms.values()) if p0 else 0),
                        "failure_code": backend.failure_code,
                    })
                    if backend.failure_code:
                        manifest["failures"].append({"condition": condition, "method_id": method["method_id"], "image_id": image.image_id, "failure_code": backend.failure_code})
                    completed += 1
                    eta = (time.perf_counter() - started) / completed * (summary["executions"] - completed)
                    print(f"[{completed}/{summary['executions']}] {condition} {method['method_id']} {image.image_id} · ETA {eta:.1f}s", flush=True)
        if manifest["failures"]:
            raise RuntimeError(f"Ablación incompleta: {len(manifest['failures'])} ejecuciones de backend fallaron")
        manifest["status"] = "completed"
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_report(run_directory, repetitions=config["metrics"]["bootstrap_repetitions"], seed=config["seed"])
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["fatal_error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        atomic_write_json(run_directory / "run_manifest.json", manifest)
    print(json.dumps({"run_id": run_id, "status": manifest["status"], "directory": str(run_directory)}, indent=2))


if __name__ == "__main__":
    main()
