from __future__ import annotations

from pathlib import Path
import sys
import unittest

import cv2
import numpy as np
from PIL import Image, ImageFilter


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import thesis_fitzpatrick.masks as masks
from thesis_fitzpatrick.masks import build_clean_skin_mask, measure_skin_colour, prepare_clean_skin_context


def reference_clean_skin_mask(image: Image.Image, lesion_mask: np.ndarray, *, lesion_margin_fraction: float = 0.015, ring_radius_fraction: float = 0.22) -> tuple[np.ndarray, dict]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    short_side = min(width, height)
    margin = max(3, round(short_side * lesion_margin_fraction))
    outer_radius = max(margin + 3, round(short_side * ring_radius_fraction))
    lesion = lesion_mask.astype(bool)
    lesion_exclusion = masks._dilate(lesion, margin)
    local_ring = masks._dilate(lesion, outer_radius) & ~lesion_exclusion
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    valid_fov = maximum > 8
    highlight = (maximum >= 248) & ((maximum - minimum) <= 22)
    highlight = masks._dilate(highlight, max(1, margin // 2))
    local_max = np.asarray(Image.fromarray(gray, mode="L").filter(ImageFilter.MaxFilter(masks._odd_size(max(2, margin)))))
    hair = ((local_max.astype(np.int16) - gray.astype(np.int16)) >= 45) & (gray <= 135)
    hair = masks._dilate(hair, max(1, margin // 3))
    clean = local_ring & valid_fov & ~highlight & ~hair
    minimum_pixels = max(256, round(width * height * 0.005))
    fallback_used = False
    if int(clean.sum()) < minimum_pixels:
        clean = valid_fov & ~lesion_exclusion & ~highlight & ~hair
        fallback_used = True
    return clean, {"lesion_margin_pixels": margin, "ring_radius_pixels": outer_radius, "fallback_to_full_complement": fallback_used, "candidate_skin_pixels": int(clean.sum())}


class CleanSkinMaskTests(unittest.TestCase):
    def test_opencv_context_matches_pil_reference_across_inputs(self) -> None:
        cases = []
        for height, width, colour in ((32, 48, (180, 135, 105)), (48, 32, (70, 45, 35)), (80, 120, (150, 105, 85))):
            rgb = np.full((height, width, 3), colour, dtype=np.uint8)
            rgb[0, :, :] = rgb[-1, :, :] = 0
            rgb[:, 0, :] = rgb[:, -1, :] = 0
            rgb[2:5, 2:5] = 255
            cv2.line(rgb, (2, height // 3), (width - 3, height // 3), (20, 20, 20), 1)
            cv2.line(rgb, (width // 3, 2), (width // 3, height - 3), (25, 25, 25), 1)
            cv2.line(rgb, (3, height - 4), (width - 4, 3), (30, 30, 30), 1)
            cv2.line(rgb, (3, height // 2), (width - 4, height // 2), (25, 25, 25), 1)
            rng = np.random.default_rng(height + width)
            noise = rng.integers(0, 5, size=rgb.shape, dtype=np.uint8)
            rgb = np.clip(rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            lesion = np.zeros((height, width), dtype=bool)
            lesion[max(1, height // 3):max(2, 2 * height // 3), max(1, width // 3):max(2, 2 * width // 3)] = True
            cases.append((Image.fromarray(rgb), lesion))
        cases.append((Image.fromarray(np.full((40, 40, 3), 150, dtype=np.uint8)), np.ones((40, 40), dtype=bool)))
        for image, lesion in cases:
            for margin_fraction, ring_fraction in ((0.01, 0.10), (0.015, 0.22), (0.04, 0.40)):
                expected, expected_metadata = reference_clean_skin_mask(image, lesion, lesion_margin_fraction=margin_fraction, ring_radius_fraction=ring_fraction)
                actual, actual_metadata = build_clean_skin_mask(image, lesion, lesion_margin_fraction=margin_fraction, ring_radius_fraction=ring_fraction)
                context = prepare_clean_skin_context(image, lesion_margin_fraction=margin_fraction, ring_radius_fraction=ring_fraction)
                reused, reused_metadata = build_clean_skin_mask(image, lesion, lesion_margin_fraction=margin_fraction, ring_radius_fraction=ring_fraction, context=context)
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(reused, expected)
                self.assertEqual(actual_metadata, expected_metadata)
                self.assertEqual(reused_metadata, expected_metadata)
                self.assertEqual(int(actual.sum()), int(expected.sum()))

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
