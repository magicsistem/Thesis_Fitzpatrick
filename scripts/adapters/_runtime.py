"""Small shared timing contract for adapter subprocesses."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time


def synchronize(torch, device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def timed_forward(torch, device: str, forward):
    warmup = int(os.environ.get("THESIS_ADAPTER_WARMUP", "0"))
    if warmup < 0:
        raise ValueError("THESIS_ADAPTER_WARMUP must be non-negative")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        with torch.inference_mode():
            forward()
    synchronize(torch, device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = forward()
    synchronize(torch, device)
    return output, (time.perf_counter() - started) * 1000, warmup


def write_metrics(torch, device: str, *, load_time_ms: float, inference_time_ms: float, warmup: int) -> None:
    value = os.environ.get("THESIS_ADAPTER_METRICS_PATH")
    if not value:
        return
    path = Path(value)
    payload = {
        "schema_version": 1, "device": device, "model_load_time_ms": load_time_ms,
        "inference_time_ms": inference_time_ms, "internal_warmup_iterations": warmup,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
