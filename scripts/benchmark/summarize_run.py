#!/usr/bin/env python3
"""Generate CSV, JSON, and Markdown summaries for an existing run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis_fitzpatrick.reporting import write_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    if not (args.run_directory / "run_manifest.json").is_file():
        raise SystemExit("run_directory no contiene run_manifest.json")
    report = write_report(args.run_directory, repetitions=args.bootstrap, seed=args.seed)
    print(json.dumps({"run_id": report["run_id"], "rows": report["rows"], "comparisons": len(report["pairwise_comparisons"])}, indent=2))


if __name__ == "__main__":
    main()
