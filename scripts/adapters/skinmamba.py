"""Run the published SkinMamba ISIC 2017/2018 checkpoints on CPU.

The official repository imports a CUDA selective-scan extension.  Inference
uses the mathematically equivalent chunked PyTorch implementation distributed
in that same repository, so no CUDA compiler or GPU is required.
"""

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


CHECKPOINT_SHA256 = {
    "isic17": "9b941f1459dd6e7660bd5ec32b2b58ba861c596ac8b14708a43a9c940f0645a7",
    "isic18": "36a2c352cd39011db3f416373be1bdd98b079d031fc778dbc640780823388543",
}
TEST_NORMALIZATION = {
    "isic17": (148.429, 25.748),
    "isic18": (149.034, 32.022),
}
IMAGE_SIZE = 224
SCAN_CHUNK_SIZE = 64


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess(image: Image.Image, dataset: str) -> np.ndarray:
    """Reproduce the official test normalization, tensor conversion and resize."""
    if dataset not in TEST_NORMALIZATION:
        raise ValueError(f"Dataset SkinMamba no soportado: {dataset}")
    values = np.asarray(image, dtype=np.float32)
    mean, standard_deviation = TEST_NORMALIZATION[dataset]
    normalized = (values - mean) / standard_deviation
    minimum = float(normalized.min())
    span = float(normalized.max()) - minimum
    if span <= 0:
        normalized = np.zeros_like(normalized, dtype=np.float32)
    else:
        normalized = (normalized - minimum) / span * 255.0
    # Pillow mode F is single-channel, so resize the three channels independently.
    channels = [
        np.asarray(
            Image.fromarray(normalized[:, :, index], mode="F").resize(
                (IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        for index in range(3)
    ]
    return np.ascontiguousarray(np.stack(channels, axis=2))


def selective_scan_cpu(
    us,
    dts,
    As,
    Bs,
    Cs,
    Ds=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
    **_kwargs,
):
    """Chunked official selective-scan recurrence implemented with PyTorch ops."""
    import torch

    if z is not None:
        raise ValueError("SkinMamba no usa el argumento z en selective_scan.")
    input_dtype = us.dtype
    if Bs.ndim == 3:
        Bs = Bs.unsqueeze(1)
    if Cs.ndim == 3:
        Cs = Cs.unsqueeze(1)
    batch, groups, state_size, sequence_length = Bs.shape
    dimensions = us.shape[1] // groups

    us = us.view(batch, groups, dimensions, sequence_length).permute(3, 0, 1, 2).float()
    dts = dts.view(batch, groups, dimensions, sequence_length).permute(3, 0, 1, 2).float()
    if delta_bias is not None:
        dts = dts + delta_bias.view(1, 1, groups, dimensions).float()
    if delta_softplus:
        dts = torch.nn.functional.softplus(dts)
    As = As.view(groups, dimensions, state_size).float()
    Bs = Bs.permute(3, 0, 1, 2).float()
    Cs = Cs.permute(3, 0, 1, 2).float()
    skip = Ds.view(groups, dimensions).float() if Ds is not None else None

    state = us.new_zeros((batch, groups, dimensions, state_size), dtype=torch.float32)
    outputs = []
    for offset in range(0, sequence_length, SCAN_CHUNK_SIZE):
        u_chunk = us[offset : offset + SCAN_CHUNK_SIZE]
        dt_chunk = dts[offset : offset + SCAN_CHUNK_SIZE]
        b_chunk = Bs[offset : offset + SCAN_CHUNK_SIZE]
        c_chunk = Cs[offset : offset + SCAN_CHUNK_SIZE]
        cumulative_time = dt_chunk.cumsum(dim=0)
        transition = torch.einsum("gdn,lbgd->lbgdn", As, cumulative_time).exp()
        delta_u_b = torch.einsum("lbgd,lbgn->lbgdn", dt_chunk * u_chunk, b_chunk)
        states = transition * (delta_u_b / transition).cumsum(dim=0)
        states = states + transition * state.unsqueeze(0)
        outputs.append(torch.einsum("lbgn,lbgdn->lbgd", c_chunk, states))
        state = states[-1]

    output = torch.cat(outputs, dim=0)
    if skip is not None:
        output = output + skip * us
    output = output.permute(1, 2, 3, 0).reshape(batch, -1, sequence_length)
    output = output.to(input_dtype)
    if return_last_state:
        return output, state.reshape(batch, -1, state_size)
    return output


def install_mamba_compatibility_module() -> None:
    root = types.ModuleType("mamba_ssm")
    ops = types.ModuleType("mamba_ssm.ops")
    interface = types.ModuleType("mamba_ssm.ops.selective_scan_interface")
    interface.selective_scan_fn = selective_scan_cpu
    interface.selective_scan_ref = selective_scan_cpu
    root.ops = ops
    ops.selective_scan_interface = interface
    sys.modules["mamba_ssm"] = root
    sys.modules["mamba_ssm.ops"] = ops
    sys.modules["mamba_ssm.ops.selective_scan_interface"] = interface


def normalize_state_dict(stored: dict) -> dict:
    """Remove THOP profiling buffers accidentally saved by the authors."""
    state_dict = stored.get("model_state_dict", stored.get("state_dict", stored))
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
        if not key.endswith(("total_ops", "total_params"))
    }


def load_model(source: Path, checkpoint: Path, dataset: str, device: str = "cpu"):
    model_file = source / "models" / "SkinMamba.py"
    if not model_file.is_file():
        raise FileNotFoundError(
            f"No se encontró el código oficial en {source}. Ejecuta scripts/setup_skinmamba.py."
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"No se encontró el checkpoint en {checkpoint}. Ejecuta scripts/setup_skinmamba.py."
        )
    expected = CHECKPOINT_SHA256[dataset]
    actual = sha256(checkpoint)
    verify_checkpoint(checkpoint, actual, expected, f"skinmamba-{dataset}")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("SkinMamba requiere PyTorch, timm y einops.") from exc

    install_mamba_compatibility_module()
    sys.path.insert(0, str(source))
    from models.SkinMamba import SkinMamba

    model = SkinMamba()
    try:
        stored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        stored = torch.load(checkpoint, map_location="cpu")
    state_dict = normalize_state_dict(stored)
    model.load_state_dict(state_dict, strict=True)
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
    result = Image.fromarray(mask, mode="L").resize(original_size, resample=Image.Resampling.NEAREST)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=sorted(CHECKPOINT_SHA256))
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
