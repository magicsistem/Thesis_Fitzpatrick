#!/usr/bin/env python3
"""Prepare leakage-safe P0: every training image uses its held-out-fold YOLO."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from thesis_fitzpatrick.benchmark import ImageInput, atomic_write_json, load_json, sha256_file, validate_dataset_manifest  # noqa: E402
from thesis_fitzpatrick.preprocessing import detector_from_config, run_p0  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path); parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--folds", required=True, type=Path); parser.add_argument("--yolo-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/benchmark/default.json")
    parser.add_argument("--artifact-root", type=Path, default=REPO_ROOT / "results/benchmark_v1/preprocessing")
    parser.add_argument("--reuse-cache", action="store_true"); parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(); manifest, folds, base_config = load_json(args.manifest), load_json(args.folds), load_json(args.config)
    validate_dataset_manifest(manifest, data_root=args.data_root, require_files=True)
    if manifest.get("split") != "train": raise SystemExit("OOF P0 accepts only split=train; sealed test is forbidden")
    if not args.confirm_run: raise SystemExit(f"OOF P0 would process {len(manifest['items'])} images; repeat with --confirm-run")
    by_id = {item["image_id"]: item for item in manifest["items"]}; seen = set(); failures = []
    for fold in folds.get("folds", []):
        frozen = args.yolo_root / f"fold-{fold['fold']}" / "frozen.json"
        if not frozen.is_file(): raise SystemExit(f"Missing frozen YOLO fold: {frozen}")
        config = copy.deepcopy(base_config); config["p0"]["yolo"]["frozen_manifest"] = str(frozen.resolve())
        detector = detector_from_config(config, REPO_ROOT)
        for image_id in fold["validation_ids"]:
            if image_id in seen: raise SystemExit(f"Fold leakage/duplication: {image_id} is held out more than once")
            seen.add(image_id); item = by_id[image_id]
            try:
                with Image.open(args.data_root / item["image_path"]) as opened: rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
                result = run_p0(ImageInput(image_id, manifest["dataset_id"], "train_oof", rgb, metadata={**item, "yolo_fold": fold["fold"]}), config, detector, cache_root=args.artifact_root, reuse_cache=args.reuse_cache)
                print(f"fold={fold['fold']} image={image_id} cache={result.cache_key}", flush=True)
            except Exception as exc: failures.append({"fold": fold["fold"], "image_id": image_id, "error": f"{type(exc).__name__}: {exc}"})
    if seen != set(by_id): raise SystemExit(f"OOF coverage mismatch: missing={len(set(by_id)-seen)} extra={len(seen-set(by_id))}")
    completion = {
        "schema_version": 1, "status": "completed" if not failures else "failed",
        "images": len(seen), "folds": len(folds.get("folds", [])), "failures": failures,
        "source_manifest_sha256": sha256_file(args.manifest), "folds_sha256": sha256_file(args.folds),
        "yolo_frozen_manifests": [
            {"fold": fold, "path": str(args.yolo_root / f"fold-{fold}" / "frozen.json"), "sha256": sha256_file(args.yolo_root / f"fold-{fold}" / "frozen.json")}
            for fold in range(5)
        ],
    }
    atomic_write_json(args.artifact_root / "oof_manifest.json", completion)
    print(json.dumps(completion, indent=2))
    if failures: raise SystemExit(2)


if __name__ == "__main__": main()
