#!/usr/bin/env python3
"""Build deterministic five-fold assignments from a disjoint train manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from thesis_fitzpatrick.benchmark import atomic_write_json, load_json  # noqa: E402
from thesis_fitzpatrick.disjoint import build_disjoint_grouped_folds  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    result = build_disjoint_grouped_folds(load_json(args.manifest), args.folds, args.seed)
    atomic_write_json(args.output, result)
    print(json.dumps({"images": sum(len(fold["validation_ids"]) for fold in result["folds"]), "validation_sizes": [len(fold["validation_ids"]) for fold in result["folds"]], "identity_fields": result["identity_fields"]}, indent=2))


if __name__ == "__main__":
    main()
