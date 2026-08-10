from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
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

    def test_headless_guard_removes_turtle_imports_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory)
            for relative_path in avit.UNUSED_GUI_IMPORTS:
                path = source / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    "import torch\nfrom turtle import forward\nVALUE = 'kept'\n",
                    encoding="utf-8",
                )

            avit.ensure_headless_source(source)
            avit.ensure_headless_source(source)

            for relative_path in avit.UNUSED_GUI_IMPORTS:
                text = (source / relative_path).read_text(encoding="utf-8")
                self.assertNotIn("from turtle import forward", text)
                self.assertIn("import torch", text)
                self.assertIn("VALUE = 'kept'", text)


if __name__ == "__main__":
    unittest.main()
