#!/usr/bin/env python3
"""Initialize, inspect, and export a versioned annotation project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis_fitzpatrick.annotations import export_consensus_manifest, initialize_project, project_status, sync_project  # noqa: E402
from thesis_fitzpatrick.benchmark import load_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    init = commands.add_parser("init"); init.add_argument("--manifest", required=True, type=Path); init.add_argument("--project", required=True, type=Path)
    status = commands.add_parser("status"); status.add_argument("--project", required=True, type=Path)
    export = commands.add_parser("export"); export.add_argument("--project", required=True, type=Path); export.add_argument("--output", required=True, type=Path)
    sync = commands.add_parser("sync"); sync.add_argument("--project", required=True, type=Path); sync.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "init": result = initialize_project(args.project, load_json(args.manifest))
    elif args.action == "status": result = project_status(args.project)
    elif args.action == "sync": result = sync_project(args.project, load_json(args.manifest))
    else: result = export_consensus_manifest(args.project, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
