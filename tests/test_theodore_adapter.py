from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "adapters"))

import theodore_isic2018


class TheodoreAdapterTests(unittest.TestCase):
    def test_preprocess_uses_zero_to_one_without_segformer_normalization(self) -> None:
        image = Image.new("RGB", (8, 8), (255, 128, 0))
        values = theodore_isic2018.preprocess(image)
        self.assertEqual(values.shape, (128, 128, 3))
        self.assertAlmostEqual(float(values[0, 0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(values[0, 0, 1]), 128 / 255, places=5)

    def test_segformer_preprocess_adds_imagenet_normalization(self) -> None:
        image = Image.new("RGB", (8, 8), (255, 128, 0))
        values = theodore_isic2018.preprocess(image, imagenet_normalized=True)
        self.assertAlmostEqual(float(values[0, 0, 0]), (1.0 - 0.485) / 0.229, places=5)
        self.assertAlmostEqual(float(values[0, 0, 2]), (0.0 - 0.406) / 0.225, places=5)

    def test_all_checkpoint_hashes_are_fixed(self) -> None:
        for digest in theodore_isic2018.CHECKPOINTS.values():
            self.assertEqual(len(digest), hashlib.sha256().digest_size * 2)

    def test_segformer_v4_keys_are_normalized_for_transformers_v5(self) -> None:
        state = {
            "model.segformer.encoder.block.0.1.attention.self.query.weight": object(),
            "model.decode_head.linear_c.2.proj.weight": object(),
        }
        expected = {
            "model.segformer.stages.0.blocks.1.attention.q_proj.weight": object(),
            "model.decode_head.linear_projections.2.proj.weight": object(),
        }
        self.assertEqual(
            set(theodore_isic2018.normalize_segformer_state_dict(state, expected)),
            set(expected),
        )


if __name__ == "__main__":
    unittest.main()
