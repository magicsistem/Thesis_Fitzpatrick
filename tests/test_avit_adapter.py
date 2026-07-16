from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "adapters"))

import avit


class AViTAdapterTests(unittest.TestCase):
    def test_preprocess_matches_imagenet_normalization(self) -> None:
        image = Image.new("RGB", (16, 8), (255, 255, 255))
        values = avit.preprocess(image)
        self.assertEqual(values.shape, (224, 224, 3))
        expected = (1.0 - np.asarray([0.485, 0.456, 0.406])) / np.asarray(
            [0.229, 0.224, 0.225]
        )
        np.testing.assert_allclose(values[0, 0], expected, rtol=1e-5)

    def test_published_checkpoint_hash_is_fixed(self) -> None:
        self.assertEqual(len(avit.CHECKPOINT_SHA256), hashlib.sha256().digest_size * 2)


if __name__ == "__main__":
    unittest.main()

