#!/usr/bin/env python3
"""Build deterministic leakage-audited folds from a training manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis_fitzpatrick.benchmark import atomic_write_json, build_grouped_folds, load_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train", choices=("train",), help="Folds can only be built from training")
    parser.add_argument("--folds", default=5, type=int)
    parser.add_argument("--seed", default=20260806, type=int)
    parser.add_argument("--data-root", type=Path, help="Defaults to data/raw/<dataset>")
    parser.add_argument("--manifest", type=Path, help="Defaults to <data-root>/manifests/<dataset>_train.json")
    parser.add_argument("--output", type=Path, help="Defaults beside the source manifest")
    parser.add_argument("--dry-run", action="store_true", help="Audit and print without writing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root or REPO_ROOT / "data" / "raw" / args.dataset
    manifest_path = args.manifest or data_root / "manifests" / f"{args.dataset}_train.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Falta el manifest de entrenamiento: {manifest_path}")
    payload = load_json(manifest_path)
    if payload.get("dataset_id") != args.dataset or payload.get("split") != "train":
        raise SystemExit("El manifest debe corresponder exactamente al dataset y split train solicitados")
    result = build_grouped_folds(payload, folds=args.folds, seed=args.seed)
    output = args.output or manifest_path.with_name(f"{args.dataset}_train_folds_{args.folds}.json")
    if not args.dry_run:
        atomic_write_json(output, result)
    summary = {
        "output": None if args.dry_run else str(output),
        "dry_run": args.dry_run,
        "images": sum(len(fold["validation_ids"]) for fold in result["folds"]),
        "groups": result["group_count"],
        "fallback_images": result["image_id_fallback_count"],
        "validation_sizes": [len(fold["validation_ids"]) for fold in result["folds"]],
        "leakage_audit": result["leakage_audit"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
