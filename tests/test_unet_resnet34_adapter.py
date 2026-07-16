from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "adapters"))

import unet_resnet34


class UNetResNet34AdapterTests(unittest.TestCase):
    def test_preprocess_matches_imagenet_normalization(self) -> None:
        image = Image.new("RGB", (12, 12), (255, 128, 0))
        values = unet_resnet34.preprocess(image)
        self.assertEqual(values.shape, (256, 256, 3))
        self.assertAlmostEqual(float(values[0, 0, 0]), (1.0 - 0.485) / 0.229, places=5)
        self.assertAlmostEqual(float(values[0, 0, 1]), (128 / 255 - 0.456) / 0.224, places=5)
        self.assertAlmostEqual(float(values[0, 0, 2]), (0.0 - 0.406) / 0.225, places=5)

    def test_published_checkpoint_hash_is_fixed(self) -> None:
        self.assertEqual(
            len(unet_resnet34.CHECKPOINT_SHA256),
            hashlib.sha256().digest_size * 2,
        )


if __name__ == "__main__":
    unittest.main()
