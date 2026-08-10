#!/usr/bin/env python3
"""Single-process CEDIA validation for the selected execution scope."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
RESOURCE_MANIFEST = REPO_ROOT / "configs" / "hpc" / "external_resources.json"
MODEL_CATALOG = REPO_ROOT / "configs" / "segmentation_models.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("code", "models", "smoke", "yolo", "b2", "benchmark"), default="code")
    parser.add_argument("--manifest", type=Path, help="Dataset manifest required by smoke/YOLO/B2/benchmark")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("THESIS_DATA_ROOT", REPO_ROOT / "data")))
    args = parser.parse_args()
    errors: list[str] = []
    warnings: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    versions = {}
    imports = {
        "numpy": "numpy", "Pillow": "PIL", "yaml": "yaml", "opencv": "cv2",
        "torch": "torch", "torchvision": "torchvision", "timm": "timm",
        "transformers": "transformers", "segmentation_models_pytorch": "segmentation_models_pytorch",
    }
    for name, module_name in imports.items():
        try:
            module = importlib.import_module(module_name)
            versions[name] = getattr(module, "__version__", "present")
        except Exception as exc:  # imports can fail through binary incompatibility
            errors.append(f"import {name}: {type(exc).__name__}: {exc}")

    torch = sys.modules.get("torch")
    cuda = bool(torch is not None and torch.cuda.is_available())
    if args.scope != "code":
        require(cuda, "CUDA is not available inside the container")
    device = None
    if cuda:
        device = torch.cuda.get_device_name(0)
        warnings.extend([] if "A100" in device else [f"GPU is not the validated A100 profile: {device}"])

    resources = json.loads(RESOURCE_MANIFEST.read_text(encoding="utf-8"))
    checkpoints = []
    for item in resources["checkpoints"]:
        path = REPO_ROOT / item["destination"]
        state = "missing"
        if path.is_file():
            state = "verified" if path.stat().st_size == item["expected_bytes"] and sha256(path) == item["sha256"] else "invalid"
        checkpoints.append({"id": item["id"], "state": state})
        required = args.scope in {"models", "smoke", "b2", "benchmark"}
        required = required or (args.scope == "yolo" and item["id"] == "darknet53-bootstrap")
        if required:
            require(state == "verified", f"checkpoint {item['id']}: {state}")
    sources = []
    for item in resources["sources"]:
        path = REPO_ROOT / item["destination"]
        state = "missing"
        if (path / ".git").is_dir():
            import subprocess
            head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True).stdout.strip()
            state = "verified" if head == item["revision"] else f"wrong_revision:{head}"
        sources.append({"id": item["id"], "state": state})
        required = args.scope in {"models", "smoke", "b2", "benchmark"}
        required = required or (args.scope == "yolo" and item["id"] == "yolov3-darknet")
        if required:
            require(state == "verified", f"source {item['id']}: {state}")

    catalog = json.loads(MODEL_CATALOG.read_text(encoding="utf-8"))["models"]
    require(len(catalog) == 15, f"catalog has {len(catalog)} neural variants instead of 15")
    require(len({item['id'] for item in catalog}) == len(catalog), "duplicate model id in catalog")

    dataset_report = None
    if args.scope in {"smoke", "yolo", "b2", "benchmark"}:
        require(args.manifest is not None, f"--manifest is mandatory for scope {args.scope}")
        if args.manifest and args.manifest.is_file():
            payload = json.loads(args.manifest.read_text(encoding="utf-8"))
            items = payload.get("items", [])
            missing = []
            for item in items:
                image = args.data_root / item["image_path"]
                if not image.is_file():
                    missing.append(str(image))
            dataset_report = {"dataset": payload.get("dataset_id"), "split": payload.get("split"), "items": len(items), "missing_images": len(missing)}
            require(not missing, f"dataset manifest has {len(missing)} missing images")
            if args.scope in {"yolo", "b2"}:
                require(payload.get("split") == "train", f"{args.scope} requires split=train, never sealed test")
        elif args.manifest:
            errors.append(f"dataset manifest missing: {args.manifest}")

    darknet = REPO_ROOT / "models" / "yolov3-darknet" / "source" / "darknet"
    if args.scope in {"yolo", "b2", "benchmark"}:
        require(os.access(darknet, os.X_OK), f"Darknet binary missing/not executable: {darknet}")
    if args.scope == "b2":
        frozen = REPO_ROOT / "results" / "benchmark_v1" / "yolo"
        require(all((frozen / f"fold-{fold}" / "frozen.json").is_file() for fold in range(5)), "B2 requires five frozen YOLO manifests")

    writable = []
    for path in (REPO_ROOT / ".cedia", REPO_ROOT / "results", args.data_root):
        writable.append({"path": str(path), "writable": path.is_dir() and os.access(path, os.W_OK)})
        require(writable[-1]["writable"], f"not writable: {path}")
    disk = shutil.disk_usage(args.data_root)
    if disk.free < 20 * 1024**3:
        warnings.append(f"Less than 20 GiB free at data root: {disk.free / 1024**3:.1f} GiB")

    report = {
        "schema_version": 1, "scope": args.scope, "passed": not errors,
        "python": sys.version.split()[0], "versions": versions,
        "cuda_available": cuda, "device": device, "catalog_models": len(catalog),
        "sources": sources, "checkpoints": checkpoints, "dataset": dataset_report,
        "writable": writable, "disk_free_gib": round(disk.free / 1024**3, 2),
        "warnings": warnings, "errors": errors,
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
