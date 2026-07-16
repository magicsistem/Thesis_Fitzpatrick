"""Run the verified U-Net/ResNet34 ISIC 2018 checkpoint on CPU."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


CHECKPOINT_SHA256 = "1ea87e341552768234b367c3b68704030bc1bc08991c323508ea5c5086d9d334"
IMAGE_SIZE = 256
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess(image: Image.Image) -> np.ndarray:
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BILINEAR)
    values = np.asarray(resized, dtype=np.float32) / 255.0
    return np.ascontiguousarray((values - IMAGENET_MEAN) / IMAGENET_STD)


def load_model(checkpoint: Path):
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"No se encontró el checkpoint en {checkpoint}. "
            "Ejecuta scripts/setup_unet_resnet34.py."
        )
    actual_sha = sha256(checkpoint)
    if actual_sha != CHECKPOINT_SHA256:
        raise ValueError(
            "El SHA-256 del checkpoint U-Net/ResNet34 no coincide: "
            f"esperado {CHECKPOINT_SHA256}, obtenido {actual_sha}."
        )
    try:
        import segmentation_models_pytorch as smp
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Instala segmentation-models-pytorch==0.5.0 en el entorno thesis-avit."
        ) from exc

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    try:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return torch, model


def infer(image_path: Path, output_path: Path, checkpoint: Path) -> None:
    torch, model = load_model(checkpoint)
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    original_size = image.size
    values = preprocess(image)
    tensor = torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
    with torch.inference_mode():
        logits = model(tensor)
        mask = (torch.sigmoid(logits)[0, 0] > 0.5).to(torch.uint8).cpu().numpy() * 255
    result = Image.fromarray(mask, mode="L").resize(original_size, resample=Image.Resampling.NEAREST)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    infer(args.image.resolve(), args.output.resolve(), args.checkpoint.resolve())


if __name__ == "__main__":
    main()
