from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis_fitzpatrick.masks import build_clean_skin_mask, measure_skin_colour


class CleanSkinMaskTests(unittest.TestCase):
    def test_excludes_lesion_black_border_and_highlight(self) -> None:
        rgb = np.full((120, 160, 3), [126, 86, 68], dtype=np.uint8)
        rgb[:8, :, :] = 0
        rgb[-8:, :, :] = 0
        rgb[:, :8, :] = 0
        rgb[:, -8:, :] = 0
        rgb[25:30, 25:30, :] = 255
        lesion = np.zeros((120, 160), dtype=bool)
        lesion[45:78, 62:98] = True

        skin, metadata = build_clean_skin_mask(Image.fromarray(rgb), lesion)

        self.assertFalse(skin[55, 80])
        self.assertFalse(skin[2, 80])
        self.assertFalse(skin[27, 27])
        self.assertGreater(int(skin.sum()), 256)
        self.assertFalse(metadata["fallback_to_full_complement"])

    def test_colour_summary_uses_only_selected_pixels(self) -> None:
        rgb = np.full((20, 20, 3), [100, 80, 60], dtype=np.uint8)
        rgb[:10, :10, :] = [200, 180, 160]
        mask = np.zeros((20, 20), dtype=bool)
        mask[10:, 10:] = True

        stats = measure_skin_colour(Image.fromarray(rgb), mask)

        self.assertEqual(stats.pixel_count, 100)
        self.assertEqual(stats.rgb_median, (100.0, 80.0, 60.0))
        self.assertAlmostEqual(stats.coverage_fraction, 0.25)


if __name__ == "__main__":
    unittest.main()
