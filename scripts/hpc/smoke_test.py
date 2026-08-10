#!/usr/bin/env python3
"""Run one real AViT checkpoint on one authorized development image and audit output."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "adapters"))
from avit import load_model, preprocess, sha256  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--image-id", help="Optional development ID; default is the first sorted manifest item")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "hpc_smoke")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("split") == "test":
        raise SystemExit("Smoke test refuses the sealed test split")
    items = sorted(manifest.get("items", []), key=lambda item: item["image_id"])
    selected = next((item for item in items if item["image_id"] == args.image_id), None) if args.image_id else (items[0] if items else None)
    if selected is None:
        raise SystemExit("Requested image is absent or the manifest is empty")
    image_path = args.data_root / selected["image_path"]
    if not image_path.is_file():
        raise SystemExit(f"Image is missing: {image_path}")
    source = REPO_ROOT / "models" / "avit" / "source"
    checkpoint = REPO_ROOT / "models" / "avit" / "checkpoints" / "AViT_ViT_B_ISIC_best.pth"

    import torch
    total_started = time.perf_counter()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA smoke requested but torch.cuda.is_available() is false")
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    load_started = time.perf_counter()
    torch, model = load_model(source, checkpoint, args.device)
    if args.device == "cuda":
        torch.cuda.synchronize()
    load_ms = (time.perf_counter() - load_started) * 1000

    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    original_size = image.size
    values = preprocess(image)
    tensor = torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0).to(device=args.device, dtype=torch.float32)
    with torch.inference_mode():
        _ = model(tensor, d="0")["seg"]
    if args.device == "cuda":
        torch.cuda.synchronize()
    samples = []
    logits = None
    for _ in range(3):
        if args.device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            logits = model(tensor, d="0")["seg"]
        if args.device == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000)
    mask = (torch.sigmoid(logits)[0, 0] >= 0.5).to(torch.uint8).cpu().numpy() * 255
    output_mask = Image.fromarray(mask, mode="L").resize(original_size, Image.Resampling.NEAREST)
    output_values = np.asarray(output_mask)
    unique = sorted(map(int, np.unique(output_values)))
    if not set(unique).issubset({0, 255}) or output_values.shape != (original_size[1], original_size[0]):
        raise SystemExit("Adapter output violates the binary/original-resolution contract")
    warning = None
    foreground_fraction = float(np.mean(output_values > 0))
    if foreground_fraction == 0 or foreground_fraction == 1:
        warning = "prediction_empty" if foreground_fraction == 0 else "prediction_saturated"

    run_root = args.output / f"avit-{selected['image_id']}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_root.mkdir(parents=True, exist_ok=False)
    output_mask.save(run_root / "lesion_mask.png")
    report = {
        "schema_version": 1, "status": "completed_with_warning" if warning else "completed",
        "scientific_result": False, "method_id": "S01", "backend_id": "avit",
        "dataset_id": manifest.get("dataset_id"), "split": manifest.get("split"), "image_id": selected["image_id"],
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "image_sha256": sha256(image_path), "checkpoint_sha256": sha256(checkpoint),
        "device": args.device, "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "input_size": list(original_size), "output_size": list(output_mask.size), "mask_values": unique,
        "foreground_fraction": foreground_fraction, "warning": warning,
        "model_load_time_ms": load_ms, "warmup_iterations": 1, "measurement_repetitions": 3,
        "inference_time_ms": {"samples": samples, "median": float(np.median(samples)), "p25": float(np.percentile(samples, 25)), "p75": float(np.percentile(samples, 75))},
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024**2 if args.device == "cuda" else None,
        "process_max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "end_to_end_time_ms": (time.perf_counter() - total_started) * 1000,
    }
    (run_root / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**report, "output": str(run_root)}, indent=2))


if __name__ == "__main__":
    main()
