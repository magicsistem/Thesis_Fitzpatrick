"""Run the verified De-LightSAM dermoscopy checkpoint on CPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image, ImageOps
from _checkpoint import verify_checkpoint


CHECKPOINT_SHA256 = "79555730abffb5b39bef5d59afb7b89b13468348b77efa144649b90fddc01778"
IMAGE_SIZE = 1024
# The official BinaryLoader accidentally uses pixel_mean as both mean and std.
OFFICIAL_PIXEL_MEAN = np.asarray((123.675, 116.280, 103.530), dtype=np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess(image: Image.Image) -> np.ndarray:
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    values = np.asarray(resized, dtype=np.float32)
    return np.ascontiguousarray((values - OFFICIAL_PIXEL_MEAN) / OFFICIAL_PIXEL_MEAN)


def load_model(source: Path, checkpoint: Path):
    if not (source / "model.py").is_file():
        raise FileNotFoundError(
            f"No se encontró De-LightSAM en {source}. Ejecuta scripts/setup_delightsam.py."
        )
    actual_sha = sha256(checkpoint)
    verify_checkpoint(checkpoint, actual_sha, CHECKPOINT_SHA256, "delightsam-dermoscopy")
    import torch

    # model.py imports cv2 but never uses it. Keep OpenCV optional for this adapter.
    if importlib.util.find_spec("cv2") is None:
        sys.modules.setdefault("cv2", types.ModuleType("cv2"))
    sys.path.insert(0, str(source))
    try:
        from model import ESPMedSAM
    finally:
        sys.path.pop(0)
    model = ESPMedSAM()
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return torch, model


def infer(image_path: Path, output_path: Path, source: Path, checkpoint: Path) -> None:
    torch, model = load_model(source, checkpoint)
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    original_size = image.size
    values = preprocess(image)
    tensor = torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0).float()
    with torch.inference_mode():
        logits, _, _ = model(x=tensor, domain_seq=0)
        mask = (torch.sigmoid(logits)[0, 0] >= 0.5).to(torch.uint8).cpu().numpy() * 255
    result = Image.fromarray(mask, mode="L").resize(original_size, Image.Resampling.NEAREST)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    infer(
        args.image.resolve(), args.output.resolve(), args.source.resolve(), args.checkpoint.resolve()
    )


if __name__ == "__main__":
    main()
