"""Run the official VM-UNet ISIC checkpoints with a PyTorch CPU scan."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageOps
from _checkpoint import verify_checkpoint
from _runtime import timed_forward, write_metrics

IMAGE_SIZE = 256
CHECKPOINTS = {
    "isic17": "61af065274210838ed65d165bd908a0b831ec6be8f2739c60e990fcfdd1071ce",
    "isic18": "a5b2c175ccb2e2fa428004a1c90023ffd80ef3a9d9c485f7f77ccbe4427abd38",
}
NORMALIZATION = {
    "isic17": (148.429, 25.748),
    "isic18": (149.034, 32.022),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess(image: Image.Image, dataset: str) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    mean, std = NORMALIZATION[dataset]
    normalized = (values - mean) / std
    low = float(normalized.min())
    high = float(normalized.max())
    if high > low:
        normalized = (normalized - low) / (high - low) * 255.0
    else:
        normalized = np.zeros_like(normalized)
    resized = Image.fromarray(normalized.astype(np.float32), mode="F") if normalized.ndim == 2 else None
    if resized is not None:
        array = np.asarray(resized.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR))
    else:
        channels = [
            np.asarray(
                Image.fromarray(normalized[:, :, channel], mode="F").resize(
                    (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            )
            for channel in range(3)
        ]
        array = np.stack(channels, axis=-1)
    return np.ascontiguousarray(array, dtype=np.float32)


def normalize_state_dict(payload: object) -> dict:
    state = payload
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                state = payload[key]
                break
    if not isinstance(state, dict):
        raise ValueError("El checkpoint VM-UNet no contiene un state_dict.")
    normalized = {}
    for key, value in state.items():
        clean = key.removeprefix("module.")
        if clean in {"total_ops", "total_params"}:
            continue
        if clean.endswith(".total_ops") or clean.endswith(".total_params"):
            continue
        normalized[clean] = value
    return normalized


def load_model(source: Path, checkpoint: Path, dataset: str, device: str = "cpu"):
    if not (source / "models" / "vmunet" / "vmunet.py").is_file():
        raise FileNotFoundError(
            f"No se encontró el código VM-UNet en {source}. Ejecuta scripts/setup_vmunet.py."
        )
    expected_sha = CHECKPOINTS[dataset]
    actual_sha = sha256(checkpoint)
    verify_checkpoint(checkpoint, actual_sha, expected_sha, f"vmunet-{dataset}")
    import torch
    from cpu_selective_scan import selective_scan_fn, selective_scan_ref

    sys.path.insert(0, str(source))
    try:
        from models.vmunet import vmamba
        from models.vmunet.vmunet import VMUNet
    finally:
        sys.path.pop(0)
    vmamba.selective_scan_fn = selective_scan_fn
    vmamba.selective_scan_ref = selective_scan_ref
    model = VMUNet(
        input_channels=3,
        num_classes=1,
        depths=[2, 2, 2, 2],
        depths_decoder=[2, 2, 2, 1],
        drop_path_rate=0.2,
        load_ckpt_path=None,
    )
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(normalize_state_dict(payload), strict=True)
    model.to(device).eval()
    return torch, model


def infer(image_path: Path, output_path: Path, source: Path, checkpoint: Path, dataset: str, device: str = "cpu") -> None:
    load_started = time.perf_counter(); torch, model = load_model(source, checkpoint, dataset, device)
    load_time_ms = (time.perf_counter() - load_started) * 1000
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    original_size = image.size
    values = preprocess(image, dataset)
    tensor = torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    output, inference_time_ms, warmup = timed_forward(torch, device, lambda: model(tensor))
    probability = output[0, 0]
    mask = (probability >= 0.5).to(torch.uint8).cpu().numpy() * 255
    write_metrics(torch, device, load_time_ms=load_time_ms, inference_time_ms=inference_time_ms, warmup=warmup)
    result = Image.fromarray(mask, mode="L").resize(original_size, Image.Resampling.NEAREST)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=sorted(CHECKPOINTS))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    infer(
        args.image.resolve(),
        args.output.resolve(),
        args.source.resolve(),
        args.checkpoint.resolve(),
        args.dataset,
        args.device,
    )


if __name__ == "__main__":
    main()
