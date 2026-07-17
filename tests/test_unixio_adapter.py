from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "adapters"))

import unixio_isic2018


class UnixioAdapterTests(unittest.TestCase):
    def test_preprocess_matches_official_imagenet_normalization(self) -> None:
        image = Image.new("RGB", (8, 8), (255, 128, 0))
        values = unixio_isic2018.preprocess(image)
        self.assertEqual(values.shape, (256, 256, 3))
        self.assertAlmostEqual(float(values[0, 0, 0]), (1.0 - 0.485) / 0.229, places=5)
        self.assertAlmostEqual(float(values[0, 0, 1]), (128 / 255 - 0.456) / 0.224, places=5)

    def test_all_checkpoint_hashes_are_fixed(self) -> None:
        for digest in unixio_isic2018.CHECKPOINTS.values():
            self.assertEqual(len(digest), hashlib.sha256().digest_size * 2)


if __name__ == "__main__":
    unittest.main()
