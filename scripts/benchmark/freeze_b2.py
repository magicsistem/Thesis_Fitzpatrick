#!/usr/bin/env python3
"""Freeze the complete 15×5 B2 fold ensemble registry with hashes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis_fitzpatrick.benchmark import atomic_write_json, load_benchmark_methods, load_json, sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", required=True, type=Path); parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--dataset", default="isic2018_task1"); parser.add_argument("--output", required=True, type=Path); parser.add_argument("--note", required=True)
    args = parser.parse_args(); identities = {}
    methods = [item for item in load_benchmark_methods(REPO_ROOT / "configs/segmentation_models.json") if item["kind"] == "neural"]
    for method in methods:
        members = []
        for fold in range(5):
            checkpoint = args.training_root / method["method_id"] / f"fold-{fold}" / f"{method['method_id']}.fold{fold}.b2.pt"
            metadata_path = checkpoint.with_suffix(checkpoint.suffix + ".metadata.json")
            if not checkpoint.is_file() or not metadata_path.is_file(): raise SystemExit(f"B2 incompleto: {method['method_id']} fold {fold}")
            metadata = load_json(metadata_path)
            digest = sha256_file(checkpoint)
            if metadata.get("protocol") != "B2" or metadata.get("method_id") != method["method_id"] or metadata.get("fold") != fold or metadata.get("checkpoint_sha256") != digest: raise SystemExit(f"Identidad B2 inválida: {metadata_path}")
            members.append({"fold": fold, "path": str(checkpoint), "sha256": digest, "metadata_path": str(metadata_path), "metadata_sha256": sha256_file(metadata_path)})
        identities[method["method_id"]] = {"aggregation": "binary_majority_vote", "members": members}
    metadata = [load_json(Path(member["metadata_path"])) for identity in identities.values() for member in identity["members"]]
    manifest_hashes = {item["source_manifest_sha256"] for item in metadata}
    folds_hashes = {item["folds_sha256"] for item in metadata}
    seeds = {item["seed"] for item in metadata}
    if len(manifest_hashes) != 1 or len(folds_hashes) != 1 or len(seeds) != 1:
        raise SystemExit("B2 no puede congelarse: manifest, folds o semilla difieren entre entrenamientos")
    payload = {"schema_version": 1, "evaluation": "B2", "frozen": True, "dataset_id": args.dataset, "base_config": str(args.base_config), "base_config_sha256": sha256_file(args.base_config), "training_protocol": {"source_manifest_sha256": next(iter(manifest_hashes)), "folds_sha256": next(iter(folds_hashes)), "seed": next(iter(seeds)), "fold_count": 5, "checkpoint_count": 75}, "b2_checkpoints": identities, "note": args.note, "frozen_utc": datetime.now(timezone.utc).isoformat()}
    atomic_write_json(args.output, payload); print(json.dumps({"output": str(args.output), "methods": len(identities), "checkpoints": sum(len(value["members"]) for value in identities.values())}, indent=2))


if __name__ == "__main__": main()
