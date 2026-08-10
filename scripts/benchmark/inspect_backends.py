#!/usr/bin/env python3
"""Load real backends in their configured environment and record parameters/resources."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))

from thesis_fitzpatrick.benchmark import load_benchmark_methods  # noqa: E402
from train_backend import arguments_from_command, load_adapter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True); parser.add_argument("--internal", action="store_true"); parser.add_argument("--output", type=Path)
    args = parser.parse_args(); methods = load_benchmark_methods(REPO_ROOT / "configs/segmentation_models.json")
    method = next((item for item in methods if item["method_id"] == args.method), None)
    if method is None: raise SystemExit("Método desconocido")
    if method["method_id"] == "S16": result = {"method_id": "S16", "parameter_count": 0, "checkpoint_bytes": 0, "device": "cpu"}
    elif not args.internal:
        adapter_python = os.environ.get("THESIS_ADAPTER_PYTHON")
        command = [adapter_python or "conda"]
        if adapter_python:
            command += [str(Path(__file__).resolve()), "--method", args.method, "--internal"]
        else:
            command += ["run", "--no-capture-output", "-n", "thesis-avit", "python", str(Path(__file__).resolve()), "--method", args.method, "--internal"]
        completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
        if completed.returncode: raise SystemExit(completed.stderr or completed.stdout)
        start = completed.stdout.rfind("\n{")
        result = json.loads(completed.stdout[start + 1:] if start >= 0 else completed.stdout)
    else:
        torch, model, _, _ = load_adapter(method, "cpu"); _, values = arguments_from_command(method); checkpoint = Path(values["checkpoint"])
        result = {"method_id": method["method_id"], "parameter_count": sum(value.numel() for value in model.parameters()), "trainable_parameter_count": sum(value.numel() for value in model.parameters() if value.requires_grad), "checkpoint_bytes": checkpoint.stat().st_size, "torch": torch.__version__, "device": "cpu"}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
