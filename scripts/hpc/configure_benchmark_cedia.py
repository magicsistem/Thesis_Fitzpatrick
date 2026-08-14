#!/usr/bin/env python3
"""Create an ignored CEDIA benchmark config pointing to a verified frozen YOLO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from thesis_fitzpatrick.benchmark import atomic_write_json, sha256_file  # noqa: E402
from thesis_fitzpatrick.hpc import artifact_identity, write_phase_state  # noqa: E402
from thesis_fitzpatrick.yolo import validate_frozen_yolo_set  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-yolo-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / ".cedia" / "benchmark.cedia.json")
    args = parser.parse_args()
    try:
        frozen = validate_frozen_yolo_set(args.frozen_yolo_root)
    except (KeyError, OSError, ValueError) as exc:
        raise SystemExit(f"Five-fold YOLO gate failed: {exc}") from exc
    frozen_manifest = args.frozen_yolo_root / "fold-0" / "frozen.json"
    contract = {"base_config_sha256": sha256_file(REPO_ROOT / "configs/benchmark/default.json"), "frozen_yolo": [{"fold": item["fold"], "manifest_sha256": item["identity_hash"]} for item in frozen]}
    state_path = args.output.with_suffix(args.output.suffix + ".phase.json")
    previous = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else None
    if previous and previous.get("status") == "completed" and previous.get("contract") == contract and args.output.is_file() and previous.get("output") == artifact_identity(args.output):
        print(args.output); return
    config = json.loads((REPO_ROOT / "configs/benchmark/default.json").read_text(encoding="utf-8"))
    config["p0"]["yolo"]["frozen_manifest"] = str(frozen_manifest.resolve())
    atomic_write_json(args.output, config)
    write_phase_state(state_path, phase="benchmark_config", status="completed", contract=contract, previous=previous, output=artifact_identity(args.output))
    print(args.output)


if __name__ == "__main__": main()
