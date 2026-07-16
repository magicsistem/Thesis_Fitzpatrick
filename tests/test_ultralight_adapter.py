from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "adapters"))

import ultralight_vm_unet as ultralight


class UltraLightAdapterTests(unittest.TestCase):
    def test_preprocess_matches_repository_minmax_normalization(self) -> None:
        pixels = np.zeros((8, 16, 3), dtype=np.uint8)
        pixels[:, :8] = 10
        pixels[:, 8:] = 110
        values = ultralight.preprocess(Image.fromarray(pixels))
        self.assertEqual(values.shape, (256, 256, 3))
        self.assertAlmostEqual(float(values.min()), 0.0)
        self.assertAlmostEqual(float(values.max()), 255.0)

    def test_published_checkpoint_hash_is_fixed(self) -> None:
        self.assertEqual(
            len(ultralight.CHECKPOINT_SHA256),
            hashlib.sha256().digest_size * 2,
        )


if __name__ == "__main__":
    unittest.main()

