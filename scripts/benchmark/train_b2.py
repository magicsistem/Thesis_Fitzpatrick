#!/usr/bin/env python3
"""Orchestrate resumable standardized B2 training for S01-S15 and five folds."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis_fitzpatrick.benchmark import atomic_write_bytes, atomic_write_json, load_benchmark_methods, load_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path); parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--folds", required=True, type=Path); parser.add_argument("--p0-root", required=True, type=Path); parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--models", default="all"); parser.add_argument("--fold", default="all", help="all or 0-4")
    parser.add_argument("--epochs", type=int, default=100); parser.add_argument("--seed", type=int, default=20260806); parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--confirm-training", action="store_true")
    parser.add_argument("--command-file", type=Path, help="With --dry-run, write one safely quoted command per line for SLURM")
    parser.add_argument("--python", dest="python_bin", help="Python executable for a persistent container environment")
    args = parser.parse_args()
    fold_payload = load_json(args.folds)
    fold_ids = [item["fold"] for item in fold_payload["folds"]] if args.fold == "all" else [int(args.fold)]
    methods = [item for item in load_benchmark_methods(REPO_ROOT / "configs" / "segmentation_models.json") if item["kind"] == "neural"]
    if args.models != "all":
        wanted = set(args.models.split(",")); methods = [item for item in methods if item["method_id"] in wanted]
        if {item["method_id"] for item in methods} != wanted: raise SystemExit("Selección B2 contiene IDs desconocidos")
    commands = []
    for method in methods:
        for fold in fold_ids:
            python_bin = args.python_bin or os.environ.get("THESIS_ADAPTER_PYTHON")
            prefix = [python_bin] if python_bin else ["conda", "run", "--no-capture-output", "-n", "thesis-avit", "python"]
            command = [*prefix, str(REPO_ROOT / "scripts/benchmark/train_backend.py"), "--method", method["method_id"], "--manifest", str(args.manifest), "--data-root", str(args.data_root), "--folds", str(args.folds), "--fold", str(fold), "--p0-root", str(args.p0_root), "--output", str(args.output / method["method_id"] / f"fold-{fold}"), "--epochs", str(args.epochs), "--seed", str(args.seed), "--device", args.device, "--confirm-training"]
            if args.resume: command.append("--resume")
            commands.append(command)
    if args.dry_run:
        if args.command_file:
            atomic_write_bytes(args.command_file, ("\n".join(shlex.join(command) for command in commands) + "\n").encode())
        print(json.dumps({"executions": len(commands), "commands": commands}, indent=2)); return
    if args.command_file: raise SystemExit("--command-file solo puede usarse con --dry-run")
    if not args.confirm_training: raise SystemExit(f"Se ejecutarían {len(commands)} entrenamientos largos. Use --dry-run y luego --confirm-training.")
    args.output.mkdir(parents=True, exist_ok=True)
    state = {"schema_version": 1, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "commands": commands, "completed": [], "failed": []}
    atomic_write_json(args.output / "b2_training_manifest.json", state)
    for command in commands:
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        key = f"{command[command.index('--method') + 1]}/fold-{command[command.index('--fold') + 1]}"
        (state["completed"] if completed.returncode == 0 else state["failed"]).append({"job": key, "returncode": completed.returncode})
        atomic_write_json(args.output / "b2_training_manifest.json", state)
    state["status"] = "completed" if not state["failed"] else "failed"
    state["finished_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(args.output / "b2_training_manifest.json", state)
    if state["failed"]: raise SystemExit(2)


if __name__ == "__main__": main()
