from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "adapters"))

import skinmamba


class SkinMambaAdapterTests(unittest.TestCase):
    def test_preprocess_matches_official_range_and_size(self) -> None:
        pixels = np.zeros((8, 16, 3), dtype=np.uint8)
        pixels[:, :8] = 10
        pixels[:, 8:] = 110
        values = skinmamba.preprocess(Image.fromarray(pixels), "isic18")
        self.assertEqual(values.shape, (224, 224, 3))
        self.assertAlmostEqual(float(values.min()), 0.0)
        self.assertAlmostEqual(float(values.max()), 255.0)

    def test_both_published_checkpoint_hashes_are_fixed(self) -> None:
        self.assertEqual(set(skinmamba.CHECKPOINT_SHA256), {"isic17", "isic18"})
        for digest in skinmamba.CHECKPOINT_SHA256.values():
            self.assertEqual(len(digest), hashlib.sha256().digest_size * 2)

    def test_checkpoint_normalization_removes_profiling_buffers(self) -> None:
        stored = {
            "module.layer.weight": object(),
            "module.layer.total_ops": object(),
            "total_params": object(),
        }
        self.assertEqual(list(skinmamba.normalize_state_dict(stored)), ["layer.weight"])


if __name__ == "__main__":
    unittest.main()
