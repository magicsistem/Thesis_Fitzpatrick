#!/usr/bin/env python3
"""Create an ignored CEDIA benchmark config pointing to a verified frozen YOLO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from thesis_fitzpatrick.benchmark import content_hash, sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-yolo", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / ".cedia" / "benchmark.cedia.json")
    args = parser.parse_args(); frozen = json.loads(args.frozen_yolo.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen" or frozen.get("architecture") != "YOLOv3-Darknet53" or not frozen.get("identity_hash"):
        raise SystemExit("The detector manifest is not a frozen YOLOv3-Darknet53 identity")
    identity = frozen.pop("identity_hash")
    if content_hash(frozen) != identity:
        raise SystemExit("The frozen YOLO identity hash is invalid")
    frozen["identity_hash"] = identity
    for field in ("cfg", "weights"):
        path = Path(frozen[f"{field}_path"])
        if not path.is_absolute(): path = REPO_ROOT / path
        if not path.is_file() or sha256_file(path) != frozen[f"{field}_sha256"]:
            raise SystemExit(f"Frozen YOLO {field} is missing or modified: {path}")
    config = json.loads((REPO_ROOT / "configs/benchmark/default.json").read_text(encoding="utf-8"))
    config["p0"]["yolo"]["frozen_manifest"] = str(args.frozen_yolo.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8"); temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__": main()
