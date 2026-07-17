from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "adapters"))

import vmunet


class VMUNetAdapterTests(unittest.TestCase):
    def test_preprocess_matches_repository_minmax_range(self) -> None:
        image = Image.new("RGB", (8, 8), (25, 100, 225))
        values = vmunet.preprocess(image, "isic18")
        self.assertEqual(values.shape, (256, 256, 3))
        self.assertAlmostEqual(float(values.min()), 0.0, places=4)
        self.assertAlmostEqual(float(values.max()), 255.0, places=4)

    def test_state_normalization_removes_profiling_buffers(self) -> None:
        state = vmunet.normalize_state_dict(
            {
                "total_ops": "root ops",
                "total_params": "root params",
                "module.layer.weight": "weight",
                "module.layer.total_ops": "nested ops",
                "module.layer.total_params": "nested params",
            }
        )
        self.assertEqual(state, {"layer.weight": "weight"})

    def test_both_checkpoint_hashes_are_fixed(self) -> None:
        for digest in vmunet.CHECKPOINTS.values():
            self.assertEqual(len(digest), hashlib.sha256().digest_size * 2)


if __name__ == "__main__":
    unittest.main()
