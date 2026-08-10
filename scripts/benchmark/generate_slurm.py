#!/usr/bin/env python3
"""Generate a portable SLURM array script for B2 or YOLO folds."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-file", required=True, type=Path, help="One already-audited command per line")
    parser.add_argument("--output", required=True, type=Path); parser.add_argument("--job-name", default="thesis-b2")
    parser.add_argument("--time", default="24:00:00"); parser.add_argument("--memory", default="32G"); parser.add_argument("--cpus", type=int, default=4); parser.add_argument("--gres")
    args = parser.parse_args()
    count = len([line for line in args.command_file.read_text(encoding="utf-8").splitlines() if line.strip()])
    if not count: raise SystemExit("command-file está vacío")
    lines = ["#!/bin/bash", f"#SBATCH --job-name={args.job_name}", f"#SBATCH --array=1-{count}", f"#SBATCH --time={args.time}", f"#SBATCH --mem={args.memory}", f"#SBATCH --cpus-per-task={args.cpus}"]
    if args.gres: lines.append(f"#SBATCH --gres={args.gres}")
    lines += ["set -euo pipefail", f"COMMAND=$(sed -n \"${{SLURM_ARRAY_TASK_ID}}p\" {str(args.command_file)!r})", "bash -lc \"$COMMAND\""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__": main()
