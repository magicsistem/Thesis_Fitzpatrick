#!/usr/bin/env python3
"""Install a user-pinned official Darknet revision and Darknet-53 bootstrap weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_DARKNET_REVISION = "f86901f6177dfc6116360a13cc06ab680e0c86b0"
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis_fitzpatrick.benchmark import atomic_write_json, sha256_file  # noqa: E402
from thesis_fitzpatrick.datasets import download_resumable  # noqa: E402


def cpu_build_command(jobs: int = 4) -> list[str]:
    """The pinned upstream Makefile enables unavailable CUDA/OpenCV by default."""
    return ["make", "GPU=0", "CUDNN=0", "OPENCV=0", f"-j{jobs}"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=OFFICIAL_DARKNET_REVISION, help="Exact official pjreddie/darknet commit")
    parser.add_argument("--source", type=Path, default=REPO_ROOT / "models/yolov3-darknet/source")
    parser.add_argument("--weights", type=Path, default=REPO_ROOT / "models/yolov3-darknet/checkpoints/darknet53.conv.74")
    parser.add_argument("--expected-weights-sha256", help="Required to publish weights as verified")
    parser.add_argument("--download-bootstrap", action="store_true"); parser.add_argument("--confirm-download", action="store_true")
    parser.add_argument("--build", action="store_true", help="Compile the pinned source using its Makefile (CPU defaults unless edited explicitly)")
    parser.add_argument("--dry-run", action="store_true", help="Show exact network/build actions without changing files")
    args = parser.parse_args()
    plan = {"revision": args.revision, "source": str(args.source), "source_present": args.source.exists(), "clone_if_missing": True, "download_bootstrap": args.download_bootstrap, "weights": str(args.weights), "build": args.build, "network_confirmation_required": True}
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return
    if not args.source.exists():
        if not args.confirm_download: raise SystemExit("Clonar Darknet requiere --confirm-download; use --dry-run para revisar el plan")
        args.source.parent.mkdir(parents=True, exist_ok=True)
        partial_source = args.source.with_name(args.source.name + ".partial")
        if not partial_source.exists():
            subprocess.run(["git", "clone", "https://github.com/pjreddie/darknet.git", str(partial_source)], check=True)
        partial_source.replace(args.source)
    revision_present = subprocess.run(["git", "cat-file", "-e", f"{args.revision}^{{commit}}"], cwd=args.source, check=False).returncode == 0
    if not revision_present:
        if not args.confirm_download: raise SystemExit("Descargar la revisión Darknet requiere --confirm-download")
        subprocess.run(["git", "fetch", "origin", args.revision], cwd=args.source, check=True)
    subprocess.run(["git", "checkout", "--detach", args.revision], cwd=args.source, check=True)
    resolved = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.source, check=True, capture_output=True, text=True).stdout.strip()
    if resolved != args.revision: raise SystemExit(f"La revisión no resolvió exactamente al commit pedido: {resolved}")
    if args.download_bootstrap:
        if not args.confirm_download: raise SystemExit("Revise tamaño/espacio y repita con --confirm-download")
        download_resumable("https://data.pjreddie.com/files/darknet53.conv.74", args.weights, args.expected_weights_sha256)
    build_command = cpu_build_command()
    if args.build: subprocess.run(build_command, cwd=args.source, check=True)
    binary = args.source / "darknet"
    payload = {"source": "https://github.com/pjreddie/darknet", "revision": resolved, "weights_url": "https://data.pjreddie.com/files/darknet53.conv.74", "weights_path": str(args.weights), "weights_sha256": sha256_file(args.weights) if args.weights.is_file() else None, "weights_verified_against_expected": bool(args.expected_weights_sha256), "build_command": build_command, "build_mode": "CPU_no_CUDA_no_system_OpenCV", "binary_path": str(binary), "binary_sha256": sha256_file(binary) if binary.is_file() else None}
    atomic_write_json(REPO_ROOT / "models/yolov3-darknet/setup_manifest.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
