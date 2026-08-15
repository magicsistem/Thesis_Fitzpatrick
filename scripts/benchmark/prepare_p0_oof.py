#!/usr/bin/env python3
"""Prepare leakage-safe P0: each train image uses its held-out-fold YOLO."""

from __future__ import annotations

import os

THREAD_LIMITS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
for _name in THREAD_LIMITS:
    os.environ[_name] = "1"

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from multiprocessing import get_context
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from thesis_fitzpatrick.benchmark import ImageInput, atomic_write_json, load_json, sha256_file, validate_dataset_manifest  # noqa: E402
from thesis_fitzpatrick.hpc import artifact_identity, write_phase_state  # noqa: E402
from thesis_fitzpatrick.preprocessing import Detector, detector_from_config, run_p0  # noqa: E402


_detector: Detector | None = None


def allocated_cpus() -> int:
    value = os.environ.get("SLURM_CPUS_PER_TASK")
    if value is None:
        return max(1, os.cpu_count() or 1)
    try:
        cpus = int(value)
    except ValueError as exc:
        raise ValueError("SLURM_CPUS_PER_TASK debe ser entero") from exc
    if cpus < 1:
        raise ValueError("SLURM_CPUS_PER_TASK debe ser positivo")
    return cpus


def resolve_workers(value: int | None, cpus: int) -> int:
    workers = max(1, math.floor(cpus * 0.75)) if value is None else value
    if not isinstance(workers, int) or not 1 <= workers <= cpus:
        raise ValueError(f"--workers debe estar entre 1 y {cpus}")
    return workers


def validate_oof_structure(manifest: dict, folds: dict, yolo_root: Path) -> tuple[dict[str, dict], list[tuple[int, list[str]]]]:
    items = manifest.get("items", [])
    by_id = {item.get("image_id"): item for item in items}
    if len(by_id) != len(items) or None in by_id:
        raise ValueError("El manifest contiene image_id duplicado o vacío")
    entries = folds.get("folds", [])
    if len(entries) != 5 or {entry.get("fold") for entry in entries} != set(range(5)):
        raise ValueError("OOF P0 requiere exactamente los folds 0..4")
    expected, assigned, plan = set(by_id), set(), []
    for entry in sorted(entries, key=lambda item: item["fold"]):
        fold = entry["fold"]
        frozen = yolo_root / f"fold-{fold}" / "frozen.json"
        frozen_payload = load_json(frozen)
        if frozen_payload.get("status") != "frozen" or frozen_payload.get("fold") != fold:
            raise ValueError(f"Frozen YOLO inválido para fold {fold}: {frozen}")
        ids = entry.get("validation_ids")
        if not isinstance(ids, list) or len(ids) != len(set(ids)):
            raise ValueError(f"validation_ids inválidos para fold {fold}")
        unknown = set(ids) - expected
        duplicate = set(ids) & assigned
        if unknown or duplicate:
            raise ValueError(f"Cobertura OOF inválida fold={fold} unknown={len(unknown)} duplicate={len(duplicate)}")
        assigned.update(ids)
        plan.append((fold, ids))
    if assigned != expected:
        raise ValueError(f"OOF coverage mismatch antes de ejecutar: missing={len(expected - assigned)} extra={len(assigned - expected)}")
    return by_id, plan


def fold_config(base_config: dict, yolo_root: Path, fold: int) -> dict:
    config = copy.deepcopy(base_config)
    config["p0"]["yolo"]["frozen_manifest"] = str((yolo_root / f"fold-{fold}" / "frozen.json").resolve())
    return config


def completion_status(completed: int, total: int, failures: list[dict[str, object]]) -> str:
    return "completed" if completed == total and not failures else "failed"


def _worker_initialize(config: dict, root: str) -> None:
    global _detector
    for name in THREAD_LIMITS:
        os.environ[name] = "1"
    cv2.setNumThreads(1)
    _detector = detector_from_config(config, Path(root))


def _worker_process(task: tuple[int, dict, str, str, str]) -> dict[str, object]:
    fold, item, data_root, artifact_root, dataset_id = task
    if _detector is None:
        raise RuntimeError("P0 worker no fue inicializado")
    image_id = item["image_id"]
    with Image.open(Path(data_root) / item["image_path"]) as opened:
        rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    result = run_p0(
        ImageInput(image_id, dataset_id, "train_oof", rgb, metadata={**item, "yolo_fold": fold}),
        _worker_config, _detector, cache_root=Path(artifact_root), reuse_cache=_worker_reuse_cache,
    )
    return {"fold": fold, "image_id": image_id, "cache_key": result.cache_key, "cache_hit": result.cache_hit}


_worker_config: dict = {}
_worker_reuse_cache = False


def _worker_initialize_with_options(config: dict, root: str, reuse_cache: bool) -> None:
    global _worker_config, _worker_reuse_cache
    _worker_config, _worker_reuse_cache = config, reuse_cache
    _worker_initialize(config, root)


def process_fold(fold: int, ids: list[str], by_id: dict[str, dict], config: dict, *, data_root: Path, artifact_root: Path, dataset_id: str, workers: int, reuse_cache: bool) -> list[dict[str, object]]:
    tasks = [(fold, by_id[image_id], str(data_root), str(artifact_root), dataset_id) for image_id in ids]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"), initializer=_worker_initialize_with_options, initargs=(config, str(REPO_ROOT), reuse_cache)) as pool:
        futures = {pool.submit(_worker_process, task): task for task in tasks}
        outcomes = []
        for future in as_completed(futures):
            task = futures[future]
            try:
                outcomes.append(future.result())
            except Exception as exc:
                outcomes.append({"fold": task[0], "image_id": task[1]["image_id"], "error": f"{type(exc).__name__}: {exc}"})
        return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path); parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--folds", required=True, type=Path); parser.add_argument("--yolo-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/benchmark/default.json")
    parser.add_argument("--artifact-root", type=Path, default=REPO_ROOT / "results/benchmark_v1/preprocessing")
    parser.add_argument("--workers", type=int); parser.add_argument("--limit", type=int); parser.add_argument("--reuse-cache", action="store_true"); parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(); manifest, folds, base_config = load_json(args.manifest), load_json(args.folds), load_json(args.config)
    validate_dataset_manifest(manifest, data_root=args.data_root, require_files=True)
    if manifest.get("split") != "train": raise SystemExit("OOF P0 accepts only split=train; sealed test is forbidden")
    try:
        cpus = allocated_cpus(); workers = resolve_workers(args.workers, cpus)
        by_id, plan = validate_oof_structure(manifest, folds, args.yolo_root)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit debe ser positivo")
    if args.limit is not None:
        remaining = args.limit; selected_plan = []
        for fold, ids in plan:
            selected = ids[:remaining]
            if selected: selected_plan.append((fold, selected))
            remaining -= len(selected)
            if not remaining: break
        plan = selected_plan
    total = sum(len(ids) for _, ids in plan)
    selected_ids = {image_id for _, ids in plan for image_id in ids}
    completion_path = args.artifact_root / ("p0_probe_manifest.json" if args.limit is not None else "oof_manifest.json")
    frozen_contract = [{"fold": fold, **artifact_identity(args.yolo_root / f"fold-{fold}" / "frozen.json")} for fold in range(5)]
    contract = {"source_manifest_sha256": sha256_file(args.manifest), "folds_sha256": sha256_file(args.folds), "base_config_sha256": sha256_file(args.config), "yolo_frozen_manifests": frozen_contract}
    state_path = args.artifact_root / ("p0_probe_phase.json" if args.limit is not None else "oof_phase.json")
    if completion_path.is_file():
        previous = load_json(completion_path)
        expected_yolo = [{"fold": fold, "path": str(args.yolo_root / f"fold-{fold}" / "frozen.json"), "sha256": sha256_file(args.yolo_root / f"fold-{fold}" / "frozen.json")} for fold in range(5)]
        valid = previous.get("status") == "completed" and previous.get("images") == total and previous.get("failures") == [] and previous.get("folds") == 5
        valid = valid and previous.get("source_manifest_sha256") == contract["source_manifest_sha256"] and previous.get("folds_sha256") == contract["folds_sha256"] and previous.get("yolo_frozen_manifests") == expected_yolo
        valid = valid and all(len(list(args.artifact_root.glob(f"*/{by_id[image_id]['image_id']}/*/preprocessing_manifest.json"))) == 1 for image_id in selected_ids)
        if valid:
            for image_id in selected_ids:
                item = by_id[image_id]
                metadata = next(args.artifact_root.glob(f"*/{item['image_id']}/*/preprocessing_manifest.json"))
                try:
                    payload = load_json(metadata)
                    if payload.get("status") != "completed" or payload.get("image_id") != item["image_id"] or payload.get("cache_key") != metadata.parent.name:
                        valid = False; break
                    artifact_identity(metadata.parent / "roi_input.png")
                except (OSError, ValueError):
                    valid = False; break
        if not valid:
            raise SystemExit(f"OOF P0 existente pero incompleto o incompatible: {completion_path}")
        print(json.dumps({**previous, "reused": True}, indent=2)); return
    if not args.confirm_run: raise SystemExit(f"OOF P0 would process {total} images; repeat with --confirm-run")
    print(f"allocated_cpus={cpus} workers={workers} target_worker_fraction={workers / cpus:.2f} opencv_threads_per_worker=1", flush=True)
    phase = write_phase_state(state_path, phase="p0_oof", status="running", contract=contract, previous=load_json(state_path) if state_path.is_file() else None)
    started = last_report = time.monotonic(); cache_hits = newly_computed = 0; failures: list[dict[str, object]] = []; completed = 0
    for fold, ids in plan:
        config = fold_config(base_config, args.yolo_root, fold)
        for outcome in process_fold(fold, ids, by_id, config, data_root=args.data_root, artifact_root=args.artifact_root, dataset_id=manifest["dataset_id"], workers=workers, reuse_cache=args.reuse_cache):
            completed += 1
            if "error" in outcome:
                failures.append(outcome)
            elif outcome["cache_hit"]:
                cache_hits += 1
            else:
                newly_computed += 1
            print(f"fold={outcome['fold']} image={outcome['image_id']} cache={outcome.get('cache_key', 'none')} cache_hit={str(bool(outcome.get('cache_hit'))).lower()}", flush=True)
            elapsed = time.monotonic() - started
            if completed == total or completed % max(1, workers) == 0 or time.monotonic() - last_report >= 10:
                rate = completed / elapsed if elapsed else 0.0; eta = (total - completed) / rate if rate else float("inf")
                print(f"progress={completed}/{total} cache_hits={cache_hits} newly_computed={newly_computed} failures={len(failures)} images_per_second={rate:.3f} eta_seconds={eta:.1f} workers={workers} allocated_cpus={cpus} target_worker_fraction={workers / cpus:.2f}", flush=True)
                last_report = time.monotonic()
    completion = {"schema_version": 1, "status": completion_status(completed, total, failures), "images": completed, "folds": 5, "failures": failures, "workers": workers, "cache_hits": cache_hits, "newly_computed": newly_computed, "probe": args.limit is not None, **contract}
    atomic_write_json(completion_path, completion)
    write_phase_state(state_path, phase="p0_oof", status=completion["status"], contract=contract, previous=phase, output=artifact_identity(completion_path), failures=len(failures))
    print(json.dumps(completion, indent=2))
    if completion["status"] != "completed": raise SystemExit(2)


if __name__ == "__main__": main()
