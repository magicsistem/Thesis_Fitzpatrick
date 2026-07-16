from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "adapters"))

import ba_transformer


class BATransformerAdapterTests(unittest.TestCase):
    def test_preprocess_is_bgr_and_zero_to_one(self) -> None:
        image = Image.new("RGB", (10, 10), (255, 128, 0))
        values = ba_transformer.preprocess(image)
        self.assertEqual(values.shape, (352, 352, 3))
        self.assertAlmostEqual(float(values[0, 0, 0]), 0.0)
        self.assertAlmostEqual(float(values[0, 0, 1]), 128 / 255)
        self.assertAlmostEqual(float(values[0, 0, 2]), 1.0)

    def test_published_checkpoint_hash_is_fixed(self) -> None:
        self.assertEqual(
            len(ba_transformer.CHECKPOINT_SHA256),
            hashlib.sha256().digest_size * 2,
        )


if __name__ == "__main__":
    unittest.main()

