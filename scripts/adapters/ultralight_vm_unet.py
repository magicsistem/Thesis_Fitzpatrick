"""Run the published UltraLight VM-UNet skin-lesion checkpoint on CPU."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import types
import time

import numpy as np
from PIL import Image, ImageOps
from _checkpoint import verify_checkpoint
from _runtime import timed_forward, write_metrics


CHECKPOINT_SHA256 = "43b11155c19c2296707ec4dee3c417529ea0b54eb111adee864b2274ec8df52a"
IMAGE_SIZE = 256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess(image: Image.Image) -> np.ndarray:
    """Reproduce the repository's per-image min-max normalization."""
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BILINEAR)
    values = np.asarray(resized, dtype=np.float32)
    minimum = float(values.min())
    span = float(values.max()) - minimum
    if span <= 0:
        return np.zeros_like(values, dtype=np.float32)
    return (values - minimum) / span * 255.0


def load_model(source: Path, checkpoint: Path, device: str = "cpu"):
    model_file = source / "models" / "UltraLight_VM_UNet.py"
    if not model_file.is_file():
        raise FileNotFoundError(
            f"No se encontró el código oficial en {source}. "
            "Ejecuta scripts/setup_ultralight_vm_unet.py."
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"No se encontró el checkpoint en {checkpoint}. "
            "Ejecuta scripts/setup_ultralight_vm_unet.py."
        )
    actual_sha = sha256(checkpoint)
    verify_checkpoint(checkpoint, actual_sha, CHECKPOINT_SHA256, "ultralight-vm-unet")

    try:
        import torch
        from cpu_mamba import Mamba
    except ImportError as exc:
        raise RuntimeError("Faltan PyTorch o las dependencias del adaptador CPU.") from exc

    compatibility_module = types.ModuleType("mamba_ssm")
    compatibility_module.Mamba = Mamba
    sys.modules["mamba_ssm"] = compatibility_module
    sys.path.insert(0, str(source))
    from models.UltraLight_VM_UNet import UltraLight_VM_UNet

    model = UltraLight_VM_UNet(
        num_classes=1,
        input_channels=3,
        c_list=[8, 16, 24, 32, 48, 64],
        split_att="fc",
        bridge=True,
    )
    try:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return torch, model


def infer(image_path: Path, output_path: Path, source: Path, checkpoint: Path, device: str = "cpu") -> None:
    load_started = time.perf_counter(); torch, model = load_model(source, checkpoint, device)
    load_time_ms = (time.perf_counter() - load_started) * 1000
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    original_size = image.size
    values = preprocess(image)
    tensor = torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    output, inference_time_ms, warmup = timed_forward(torch, device, lambda: model(tensor))
    probability = output[0, 0]
    mask = (probability >= 0.5).to(torch.uint8).cpu().numpy() * 255
    write_metrics(torch, device, load_time_ms=load_time_ms, inference_time_ms=inference_time_ms, warmup=warmup)
    result = Image.fromarray(mask, mode="L").resize(original_size, resample=Image.Resampling.NEAREST)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    infer(
        args.image.resolve(),
        args.output.resolve(),
        args.source.resolve(),
        args.checkpoint.resolve(),
        args.device,
    )


if __name__ == "__main__":
    main()
