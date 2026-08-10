"""Run the official Boundary-aware Transformer ISIC 2016 checkpoint on CPU."""

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


CHECKPOINT_SHA256 = "62b4148b26b01b0b17b4125d74115ed49c507eab4d39ec8ff8e063a2f7980233"
IMAGE_SIZE = 352
AUTHOR_RESNET50_PATH = "/home/wjc/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess(image: Image.Image) -> np.ndarray:
    """Match the repository's OpenCV resize, BGR order, and 0–1 scale."""
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BICUBIC)
    rgb = np.asarray(resized, dtype=np.float32)
    return np.ascontiguousarray(rgb[:, :, ::-1]) / 255.0


def _blank_resnet50_state_dict() -> dict:
    """Build a shape-compatible bootstrap state dict without a download."""
    from torchvision.models import resnet50

    return resnet50(weights=None).state_dict()


def _construct_model(torch, model_class):
    """Construct BAT despite the author's machine-specific ResNet path.

    The official constructor insists on loading an absolute cache path from the
    author's workstation. A blank, shape-compatible ResNet state is sufficient
    during construction because the verified full BA checkpoint is loaded
    strictly immediately afterwards.
    """
    original_load = torch.load

    def portable_load(path, *args, **kwargs):
        if str(path) == AUTHOR_RESNET50_PATH:
            return _blank_resnet50_state_dict()
        return original_load(path, *args, **kwargs)

    torch.load = portable_load
    try:
        return model_class(
            num_classes=1,
            num_layers=50,
            point_pred=1,
            decoder=True,
            transformer_type_index=0,
        )
    finally:
        torch.load = original_load


def _normalize_checkpoint_keys(state_dict: dict) -> dict:
    """Remove the DataParallel prefix exactly as the official loader does."""
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def load_model(source: Path, checkpoint: Path, device: str = "cpu"):
    if not (source / "Ours" / "Base_transformer.py").is_file():
        raise FileNotFoundError(
            f"No se encontró el código oficial en {source}. "
            "Ejecuta scripts/setup_ba_transformer.py."
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"No se encontró el checkpoint en {checkpoint}. "
            "Ejecuta scripts/setup_ba_transformer.py."
        )
    actual_sha = sha256(checkpoint)
    verify_checkpoint(checkpoint, actual_sha, CHECKPOINT_SHA256, "ba-transformer")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("BA-Transformer necesita el entorno thesis-avit con PyTorch.") from exc
    sys.path.insert(0, str(source))
    from Ours.Base_transformer import BAT

    model = _construct_model(torch, BAT)
    try:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(_normalize_checkpoint_keys(state_dict), strict=True)
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
    logits, _ = output
    mask = (torch.sigmoid(logits)[0, 0] >= 0.5).to(torch.uint8).cpu().numpy() * 255
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
