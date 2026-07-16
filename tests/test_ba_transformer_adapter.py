from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

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

    def test_constructor_bypasses_author_absolute_resnet_path(self) -> None:
        original_load = Mock(return_value={"checkpoint": "loaded"})

        class FakeTorch:
            load = original_load

        class FakeBAT:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs
                self.bootstrap = FakeTorch.load(ba_transformer.AUTHOR_RESNET50_PATH)

        bootstrap = {"portable": "resnet50"}
        with patch.object(
            ba_transformer,
            "_blank_resnet50_state_dict",
            return_value=bootstrap,
        ):
            model = ba_transformer._construct_model(FakeTorch, FakeBAT)

        self.assertEqual(model.bootstrap, bootstrap)
        self.assertEqual(model.kwargs["num_layers"], 50)
        self.assertIs(FakeTorch.load, original_load)
        self.assertEqual(FakeTorch.load("real-checkpoint.pkl"), {"checkpoint": "loaded"})

    def test_checkpoint_normalization_removes_data_parallel_prefix(self) -> None:
        marker = object()
        normalized = ba_transformer._normalize_checkpoint_keys({
            "module.deeplab.resnet.0.weight": marker,
            "query_positions": marker,
        })
        self.assertEqual(
            set(normalized),
            {"deeplab.resnet.0.weight", "query_positions"},
        )
        self.assertIs(normalized["deeplab.resnet.0.weight"], marker)


if __name__ == "__main__":
    unittest.main()
