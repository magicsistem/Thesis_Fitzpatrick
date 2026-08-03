"""Run the official AViT ISIC 2018 checkpoint on one image."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps


CHECKPOINT_SHA256 = "9b4ad401483d96535769433f4781da42179bec6a7ef932a1d02a7f786e9f24db"
IMAGE_SIZE = 224


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess(image: Image.Image) -> np.ndarray:
    """Match the official AViT evaluation transform without albumentations."""
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BILINEAR)
    values = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    return (values - mean) / std


def load_model(source: Path, checkpoint: Path, device: str):
    if not (source / "Models" / "Transformer" / "ViT_adapters.py").is_file():
        raise FileNotFoundError(
            f"No se encontró el código oficial de AViT en {source}. "
            "Ejecuta scripts/setup_avit_model.py."
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"No se encontró el checkpoint de AViT en {checkpoint}. "
            "Ejecuta scripts/setup_avit_model.py."
        )
    actual_sha = sha256(checkpoint)
    if actual_sha != CHECKPOINT_SHA256:
        raise ValueError(
            "El SHA-256 del checkpoint AViT no coincide: "
            f"esperado {CHECKPOINT_SHA256}, obtenido {actual_sha}."
        )

    sys.path.insert(0, str(source))
    try:
        import torch
        from Models.Transformer.ViT_adapters import ViTSeg_CNNprompt_adapt
    except ImportError as exc:
        raise RuntimeError(
            "Falta una dependencia de AViT. Crea el entorno con "
            "configs/environments/avit-cpu.yml."
        ) from exc

    model = ViTSeg_CNNprompt_adapt(
        pretrained=False,
        pretrained_vit_name="vit_base_patch16_224_in21k",
        pretrained_folder="",
        img_size=IMAGE_SIZE,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        debug=False,
        adapt_method="MLP",
        num_domains=1,
    )
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    state_dict = payload.get("model_weights", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return torch, model


def infer(image_path: Path, output_path: Path, source: Path, checkpoint: Path, device: str) -> None:
    if device == "cuda":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    torch, model = load_model(source, checkpoint, device)
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    original_size = image.size
    values = preprocess(image)
    tensor = torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0).to(
        device=device, dtype=torch.float32
    )
    with torch.inference_mode():
        logits = model(tensor, d="0")["seg"]
        mask = (torch.sigmoid(logits)[0, 0] >= 0.5).to(torch.uint8).cpu().numpy() * 255
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
