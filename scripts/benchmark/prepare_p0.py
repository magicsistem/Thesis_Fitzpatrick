#!/usr/bin/env python3
"""Precompute the shared P0 once per manifest image without running a backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis_fitzpatrick.benchmark import ImageInput, load_json, validate_dataset_manifest  # noqa: E402
from thesis_fitzpatrick.preprocessing import detector_from_config, run_p0  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path); parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/benchmark/default.json")
    parser.add_argument("--artifact-root", type=Path, default=REPO_ROOT / "results/benchmark_v1/preprocessing")
    parser.add_argument("--limit", type=int); parser.add_argument("--reuse-cache", action="store_true"); parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(); manifest = load_json(args.manifest)
    validate_dataset_manifest(manifest, data_root=args.data_root, require_files=True)
    items = manifest["items"][:args.limit] if args.limit else manifest["items"]
    if len(items) > 5 and not args.confirm_run: raise SystemExit(f"P0 procesaría {len(items)} imágenes; repita con --confirm-run.")
    config = load_json(args.config); detector = detector_from_config(config, REPO_ROOT)
    failures = []
    for index, item in enumerate(items, 1):
        try:
            with Image.open(args.data_root / item["image_path"]) as opened: rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
            result = run_p0(ImageInput(item["image_id"], manifest["dataset_id"], manifest["split"], rgb, metadata=item), config, detector, cache_root=args.artifact_root, reuse_cache=args.reuse_cache)
            print(f"[{index}/{len(items)}] {item['image_id']} {result.cache_key}", flush=True)
        except Exception as exc: failures.append({"image_id": item["image_id"], "error": f"{type(exc).__name__}: {exc}"})
    print(json.dumps({"processed": len(items) - len(failures), "failures": failures}, indent=2))
    if failures: raise SystemExit(2)


if __name__ == "__main__": main()
