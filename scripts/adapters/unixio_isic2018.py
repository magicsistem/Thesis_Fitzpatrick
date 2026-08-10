"""Run verified Unixio/BertinAm U-Net variants trained on ISIC 2018."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps
from _checkpoint import verify_checkpoint
from _runtime import timed_forward, write_metrics


IMAGE_SIZE = 256
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
CHECKPOINTS = {
    "unet": "99bf0a32d36ac28a42db1299af6a0ad4d562efdf685cf3fcdccd3ae8f94d1448",
    "unetpp": "d2ba5876ffd449f426205d8498b1165618ee0159a286bfa701ff2492544ca269",
    "attention_unet": "0d41ac498ac44d2b07ea88ef75f92370ec5312d7e6d32c39f9f0a0ce12b4812b",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess(image: Image.Image) -> np.ndarray:
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    values = np.asarray(resized, dtype=np.float32) / 255.0
    return np.ascontiguousarray((values - IMAGENET_MEAN) / IMAGENET_STD)


def build_attention_unet(torch):
    nn = torch.nn

    class ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)

    class AttentionGate(nn.Module):
        def __init__(self, f_g: int, f_x: int, f_int: int):
            super().__init__()
            self.W_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1, bias=True), nn.BatchNorm2d(f_int))
            self.W_x = nn.Sequential(nn.Conv2d(f_x, f_int, 1, bias=True), nn.BatchNorm2d(f_int))
            self.psi = nn.Sequential(
                nn.Conv2d(f_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid()
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, g, x):
            return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))

    class AttentionUNet(nn.Module):
        def __init__(self):
            super().__init__()
            f1, f2, f3, f4 = (64, 128, 256, 512)
            self.pool = nn.MaxPool2d(2)
            self.enc1 = ConvBlock(3, f1)
            self.enc2 = ConvBlock(f1, f2)
            self.enc3 = ConvBlock(f2, f3)
            self.enc4 = ConvBlock(f3, f4)
            self.bottleneck = ConvBlock(f4, f4 * 2)
            self.up4 = nn.ConvTranspose2d(f4 * 2, f4, 2, stride=2)
            self.att4 = AttentionGate(f4, f4, f4 // 2)
            self.dec4 = ConvBlock(f4 * 2, f4)
            self.up3 = nn.ConvTranspose2d(f4, f3, 2, stride=2)
            self.att3 = AttentionGate(f3, f3, f3 // 2)
            self.dec3 = ConvBlock(f3 * 2, f3)
            self.up2 = nn.ConvTranspose2d(f3, f2, 2, stride=2)
            self.att2 = AttentionGate(f2, f2, f2 // 2)
            self.dec2 = ConvBlock(f2 * 2, f2)
            self.up1 = nn.ConvTranspose2d(f2, f1, 2, stride=2)
            self.att1 = AttentionGate(f1, f1, f1 // 2)
            self.dec1 = ConvBlock(f1 * 2, f1)
            self.final = nn.Conv2d(f1, 1, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            bottleneck = self.bottleneck(self.pool(e4))
            d4 = self.up4(bottleneck)
            d4 = self.dec4(torch.cat((self.att4(d4, e4), d4), dim=1))
            d3 = self.up3(d4)
            d3 = self.dec3(torch.cat((self.att3(d3, e3), d3), dim=1))
            d2 = self.up2(d3)
            d2 = self.dec2(torch.cat((self.att2(d2, e2), d2), dim=1))
            d1 = self.up1(d2)
            d1 = self.dec1(torch.cat((self.att1(d1, e1), d1), dim=1))
            return self.final(d1)

    return AttentionUNet()


def build_model(variant: str, torch):
    if variant == "attention_unet":
        return build_attention_unet(torch)
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise RuntimeError(
            "Instala segmentation-models-pytorch==0.5.0 en thesis-avit."
        ) from exc
    constructor = smp.Unet if variant == "unet" else smp.UnetPlusPlus
    return constructor(
        encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1
    )


def load_model(checkpoint: Path, variant: str, device: str = "cpu"):
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"No se encontró {checkpoint}. Ejecuta scripts/setup_unixio_isic2018.py."
        )
    expected_sha = CHECKPOINTS[variant]
    actual_sha = sha256(checkpoint)
    verify_checkpoint(checkpoint, actual_sha, expected_sha, f"unixio-{variant.replace('_', '-')}-isic2018")
    import torch

    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    if "model_state" not in payload:
        raise ValueError("El checkpoint Unixio no contiene model_state.")
    model = build_model(variant, torch)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    return torch, model


def infer(image_path: Path, output_path: Path, checkpoint: Path, variant: str, device: str = "cpu") -> None:
    load_started = time.perf_counter(); torch, model = load_model(checkpoint, variant, device)
    load_time_ms = (time.perf_counter() - load_started) * 1000
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    original_size = image.size
    values = preprocess(image)
    tensor = torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    logits, inference_time_ms, warmup = timed_forward(torch, device, lambda: model(tensor))
    probability = torch.sigmoid(logits)
    probability = torch.nn.functional.interpolate(
        probability, size=(original_size[1], original_size[0]), mode="bilinear", align_corners=False
    )
    mask = (probability[0, 0] > 0.5).to(torch.uint8).cpu().numpy() * 255
    write_metrics(torch, device, load_time_ms=load_time_ms, inference_time_ms=inference_time_ms, warmup=warmup)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=sorted(CHECKPOINTS))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    infer(args.image.resolve(), args.output.resolve(), args.checkpoint.resolve(), args.variant, args.device)


if __name__ == "__main__":
    main()
