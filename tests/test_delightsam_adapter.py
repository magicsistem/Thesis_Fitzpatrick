from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "adapters"))

import delightsam


class DeLightSAMAdapterTests(unittest.TestCase):
    def test_preprocess_preserves_official_loader_normalization(self) -> None:
        image = Image.new("RGB", (8, 8), (255, 128, 0))
        values = delightsam.preprocess(image)
        self.assertEqual(values.shape, (1024, 1024, 3))
        self.assertAlmostEqual(float(values[0, 0, 0]), (255 - 123.675) / 123.675, places=5)
        self.assertAlmostEqual(float(values[0, 0, 2]), -1.0, places=5)

    def test_checkpoint_hash_is_fixed(self) -> None:
        self.assertEqual(
            len(delightsam.CHECKPOINT_SHA256), hashlib.sha256().digest_size * 2
        )


if __name__ == "__main__":
    unittest.main()
