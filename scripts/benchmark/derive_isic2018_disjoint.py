#!/usr/bin/env python3
"""Derive, audit and publish leakage-free ISIC 2018 development manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis_fitzpatrick.benchmark import atomic_write_json, load_json  # noqa: E402
from thesis_fitzpatrick.datasets import audit_manifest_overlap  # noqa: E402
from thesis_fitzpatrick.disjoint import derive_disjoint_isic2018  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-train-count", type=int)
    parser.add_argument("--expected-validation-count", type=int)
    parser.add_argument("--expected-test-count", type=int)
    args = parser.parse_args()

    train, validation, test = map(load_json, (args.train, args.validation, args.test))
    train_disjoint, validation_disjoint, report = derive_disjoint_isic2018(train, validation, test)
    actual = {
        "train": len(train_disjoint["items"]),
        "validation": len(validation_disjoint["items"]),
        "test": len(test["items"]),
    }
    expected = {
        "train": args.expected_train_count,
        "validation": args.expected_validation_count,
        "test": args.expected_test_count,
    }
    mismatches = {split: {"expected": count, "actual": actual[split]} for split, count in expected.items() if count is not None and actual[split] != count}
    if mismatches:
        raise SystemExit("Unexpected disjoint counts: " + json.dumps(mismatches, sort_keys=True))

    audit = audit_manifest_overlap([train_disjoint, validation_disjoint, test])
    if not audit["passed"]:
        raise SystemExit(f"Disjoint derivation still has {len(audit['collisions'])} collisions")
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_root / "isic2018_task1_train_disjoint.json", train_disjoint)
    atomic_write_json(args.output_root / "isic2018_task1_validation_disjoint.json", validation_disjoint)
    atomic_write_json(args.output_root / "isic2018_task1_disjoint_derivation_report.json", report)
    atomic_write_json(args.output_root / "isic2018_task1_disjoint_overlap_audit.json", audit)
    print(json.dumps({"counts": actual, "audit_passed": True, "output_root": str(args.output_root)}, indent=2))


if __name__ == "__main__":
    main()
