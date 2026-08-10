"""Run the three verified Theodore Ioannidis ISIC 2018 models on CPU."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image, ImageOps
from _checkpoint import verify_checkpoint
from _runtime import timed_forward, write_metrics


IMAGE_SIZE = 128
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
CHECKPOINTS = {
    "unet": "d58601e6433b7ebeab5ec4249ca02e413b62b28cd9e69b1719a368cb6deae5bb",
    "inception": "18c17a1d87be5906b31a76558132e3c3fc16e643747b8e0859c25cb914eadce9",
    "segformer": "0dcb4e5c9d19ab4caffaa324e4cf26090bcc9d37fb47840e38d81268691d8341",
}


def normalize_segformer_state_dict(state: dict, expected: dict) -> dict:
    """Translate Transformers 4.x SegFormer keys to their 5.x names."""
    if state.keys() == expected.keys():
        return state

    normalized = {}
    for key, value in state.items():
        key = re.sub(
            r"\.segformer\.encoder\.patch_embeddings\.(\d+)\.",
            r".segformer.stages.\1.patch_embeddings.",
            key,
        )
        key = re.sub(
            r"\.segformer\.encoder\.block\.(\d+)\.(\d+)\.",
            r".segformer.stages.\1.blocks.\2.",
            key,
        )
        key = re.sub(
            r"\.segformer\.encoder\.layer_norm\.(\d+)\.",
            r".segformer.stages.\1.layer_norm.",
            key,
        )
        key = re.sub(
            r"\.decode_head\.linear_c\.(\d+)\.",
            r".decode_head.linear_projections.\1.",
            key,
        )
        for old, new in (
            (".layer_norm_1.", ".layernorm_before."),
            (".attention.self.query.", ".attention.q_proj."),
            (".attention.self.key.", ".attention.k_proj."),
            (".attention.self.value.", ".attention.v_proj."),
            (".attention.self.sr.", ".attention.sequence_reduction.sequence_reduction."),
            (".attention.self.layer_norm.", ".attention.sequence_reduction.layer_norm."),
            (".attention.output.dense.", ".attention.o_proj."),
            (".layer_norm_2.", ".layernorm_after."),
            (".mlp.dense1.", ".mlp.fc1."),
            (".mlp.dense2.", ".mlp.fc2."),
        ):
            key = key.replace(old, new)
        normalized[key] = value
    return normalized


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess(image: Image.Image, *, imagenet_normalized: bool = False) -> np.ndarray:
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BILINEAR)
    values = np.asarray(resized, dtype=np.float32) / 255.0
    if imagenet_normalized:
        values = (values - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(values)


def build_model(variant: str, torch):
    nn = torch.nn
    functional = torch.nn.functional

    def conv_block(in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder1 = conv_block(3, 64)
            self.pool1 = nn.MaxPool2d(2)
            self.encoder2 = conv_block(64, 128)
            self.pool2 = nn.MaxPool2d(2)
            self.encoder3 = conv_block(128, 256)
            self.pool3 = nn.MaxPool2d(2)
            self.bottleneck = conv_block(256, 512)
            self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.decoder3 = conv_block(512, 256)
            self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.decoder2 = conv_block(256, 128)
            self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.decoder1 = conv_block(128, 64)
            self.final = nn.Conv2d(64, 2, kernel_size=1)

        def forward(self, x):
            enc1 = self.encoder1(x)
            enc2 = self.encoder2(self.pool1(enc1))
            enc3 = self.encoder3(self.pool2(enc2))
            bottleneck = self.bottleneck(self.pool3(enc3))
            dec3 = self.decoder3(torch.cat((self.upconv3(bottleneck), enc3), dim=1))
            dec2 = self.decoder2(torch.cat((self.upconv2(dec3), enc2), dim=1))
            dec1 = self.decoder1(torch.cat((self.upconv1(dec2), enc1), dim=1))
            return self.final(dec1)

    class InceptionBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.b1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.ReLU(inplace=True)
            )
            self.b2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.b3 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
            )
            self.b4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return torch.cat((self.b1(x), self.b2(x), self.b3(x), self.b4(x)), dim=1)

    class Inception(nn.Module):
        def __init__(self):
            super().__init__()
            self.inception1 = InceptionBlock(3, 64)
            self.inception2 = InceptionBlock(256, 128)
            self.inception3 = InceptionBlock(512, 256)
            self.conv1x1 = nn.Conv2d(1024, 2, kernel_size=1)

        def forward(self, x):
            height, width = x.shape[2:]
            x = self.inception1(x)
            x = self.inception2(x)
            x = self.inception3(x)
            return functional.interpolate(
                self.conv1x1(x), size=(height, width), mode="bilinear", align_corners=True
            )

    if variant == "unet":
        return UNet()
    if variant == "inception":
        return Inception()
    if variant == "segformer":
        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except ImportError as exc:
            raise RuntimeError("Instala transformers en el entorno thesis-avit.") from exc

        class Segformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = SegformerForSemanticSegmentation(SegformerConfig(num_labels=2))

            def forward(self, x):
                logits = self.model(pixel_values=x).logits
                return functional.interpolate(
                    logits, size=x.shape[2:], mode="bilinear", align_corners=True
                )

        return Segformer()
    raise ValueError(f"Variante Theodore desconocida: {variant}")


def load_model(checkpoint: Path, variant: str, device: str = "cpu"):
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"No se encontró {checkpoint}. Ejecuta scripts/setup_theodore_isic2018.py."
        )
    expected_sha = CHECKPOINTS[variant]
    actual_sha = sha256(checkpoint)
    verify_checkpoint(checkpoint, actual_sha, expected_sha, f"theodore-{variant.replace('_', '-')}-isic2018")
    import torch

    model = build_model(variant, torch)
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    if variant == "segformer":
        state = normalize_segformer_state_dict(state, model.state_dict())
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return torch, model


def infer(image_path: Path, output_path: Path, checkpoint: Path, variant: str, device: str = "cpu") -> None:
    load_started = time.perf_counter(); torch, model = load_model(checkpoint, variant, device)
    load_time_ms = (time.perf_counter() - load_started) * 1000
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    original_size = image.size
    values = preprocess(image, imagenet_normalized=variant == "segformer")
    tensor = torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    logits, inference_time_ms, warmup = timed_forward(torch, device, lambda: model(tensor))
    mask = torch.argmax(logits[0], dim=0).to(torch.uint8).cpu().numpy() * 255
    write_metrics(torch, device, load_time_ms=load_time_ms, inference_time_ms=inference_time_ms, warmup=warmup)
    result = Image.fromarray(mask, mode="L").resize(original_size, Image.Resampling.NEAREST)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)


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
