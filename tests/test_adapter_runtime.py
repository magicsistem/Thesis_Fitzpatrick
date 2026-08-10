from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts/adapters"))
from _runtime import timed_forward, write_metrics


class _InferenceMode:
    def __enter__(self): return self
    def __exit__(self, *_): return False


class _Cuda:
    def __init__(self): self.syncs = 0
    def synchronize(self): self.syncs += 1
    def reset_peak_memory_stats(self): pass
    def max_memory_allocated(self): return 12 * 1024**2


class _Torch:
    def __init__(self): self.cuda = _Cuda()
    def inference_mode(self): return _InferenceMode()


class AdapterRuntimeTests(unittest.TestCase):
    def test_warmup_and_measurement_share_the_same_loaded_process(self):
        torch = _Torch(); calls = []
        with patch.dict(os.environ, {"THESIS_ADAPTER_WARMUP": "2"}):
            result, elapsed, warmup = timed_forward(torch, "cuda", lambda: calls.append(len(calls)) or "mask")
        self.assertEqual((result, warmup, len(calls)), ("mask", 2, 3))
        self.assertGreaterEqual(elapsed, 0); self.assertEqual(torch.cuda.syncs, 2)

    def test_sidecar_records_vram_and_separate_times_atomically(self):
        torch = _Torch()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            with patch.dict(os.environ, {"THESIS_ADAPTER_METRICS_PATH": str(path)}):
                write_metrics(torch, "cuda", load_time_ms=2.0, inference_time_ms=3.0, warmup=1)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["peak_vram_mb"], 12)
        self.assertEqual(payload["model_load_time_ms"], 2.0)
        self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__": unittest.main()
