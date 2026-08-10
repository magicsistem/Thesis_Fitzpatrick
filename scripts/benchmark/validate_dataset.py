#!/usr/bin/env python3
"""Validate an explicitly registered dataset manifest without modifying data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis_fitzpatrick.benchmark import load_json, validate_dataset_manifest, validate_dataset_registry  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Registered dataset id, for example isic2018_task1")
    parser.add_argument("--data-root", required=True, type=Path, help="Dataset root; manifest paths are resolved below it")
    parser.add_argument("--manifest", type=Path, help="Versioned JSON manifest (otherwise manifests/<dataset>_*.json)")
    parser.add_argument("--split", choices=("train", "validation", "test"), help="Require one split")
    parser.add_argument("--allow-count-mismatch", action="store_true", help="Report rather than reject non-official item counts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = validate_dataset_registry(load_json(REPO_ROOT / "configs" / "benchmark" / "datasets.json"))
    dataset = next((item for item in registry if item["id"] == args.dataset), None)
    if dataset is None:
        raise SystemExit(f"Dataset no registrado: {args.dataset}")
    if not dataset["enabled"]:
        raise SystemExit(f"Dataset deshabilitado: {dataset['warning']}")
    if args.manifest:
        manifests = [args.manifest]
    else:
        pattern = f"{args.dataset}_{args.split}.json" if args.split else f"{args.dataset}_*.json"
        manifests = sorted((args.data_root / "manifests").glob(pattern))
    if not manifests:
        expected = args.manifest or args.data_root / "manifests" / f"{args.dataset}_{args.split or '<split>'}.json"
        raise SystemExit(f"No se encontró un manifest explícito. Ruta esperada: {expected}")
    audits = []
    for path in manifests:
        payload = load_json(path)
        if payload.get("dataset_id") != args.dataset or (args.split and payload.get("split") != args.split):
            raise SystemExit(f"Identidad dataset/split incompatible en {path}")
        audit = validate_dataset_manifest(payload, data_root=args.data_root, require_files=True)
        expected_count = dataset.get("expected_counts", {}).get(payload["split"])
        audit["manifest"] = str(path)
        audit["expected_items"] = expected_count
        audit["count_matches"] = expected_count is None or expected_count == audit["items"]
        if not audit["count_matches"] and not args.allow_count_mismatch:
            raise SystemExit(f"{path}: se esperaban {expected_count} elementos y hay {audit['items']}")
        audits.append(audit)
    print(json.dumps({"dataset": dataset, "audits": audits}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
