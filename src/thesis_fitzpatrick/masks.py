"""Post-process lesion masks and measure surrounding skin colour robustly."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class SkinColorStats:
    pixel_count: int
    coverage_fraction: float
    rgb_median: tuple[float, float, float]
    rgb_trimmed_mean: tuple[float, float, float]
    lab_median: tuple[float, float, float]
    ita_degrees: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CleanSkinContext:
    rgb: np.ndarray
    gray: np.ndarray
    valid_fov: np.ndarray
    highlight: np.ndarray
    hair: np.ndarray
    width: int
    height: int
    margin: int
    outer_radius: int


def _odd_size(radius: int) -> int:
    return max(3, radius * 2 + 1) | 1


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    # Pillow's MaxFilter cost grows quickly with very large kernels. For wide
    # contextual rings, perform the operation on a nearest-neighbour reduced
    # binary mask and expand it again. Boundary precision is governed by the
    # small margin dilation, which stays at full resolution.
    if radius > 24:
        scale = int(np.ceil(radius / 24))
        reduced_size = (
            max(1, int(np.ceil(image.width / scale))),
            max(1, int(np.ceil(image.height / scale))),
        )
        reduced = image.resize(reduced_size, resample=Image.Resampling.NEAREST)
        reduced_radius = int(np.ceil(radius / scale))
        dilated = reduced.filter(ImageFilter.MaxFilter(_odd_size(reduced_radius)))
        dilated = dilated.resize(image.size, resample=Image.Resampling.NEAREST)
        return np.asarray(dilated) > 0
    return np.asarray(image.filter(ImageFilter.MaxFilter(_odd_size(radius)))) > 0


def normalize_binary_mask(mask: Image.Image, size: tuple[int, int]) -> np.ndarray:
    gray = mask.convert("L")
    if gray.size != size:
        gray = gray.resize(size, resample=Image.Resampling.NEAREST)
    return np.asarray(gray, dtype=np.uint8) >= 128


def prepare_clean_skin_context(
    image: Image.Image,
    *,
    lesion_margin_fraction: float = 0.015,
    ring_radius_fraction: float = 0.22,
) -> CleanSkinContext:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    short_side = min(width, height)
    margin = max(3, round(short_side * lesion_margin_fraction))
    outer_radius = max(margin + 3, round(short_side * ring_radius_fraction))
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    valid_fov = maximum > 8
    highlight = (maximum >= 248) & ((maximum - minimum) <= 22)
    highlight = _dilate(highlight, max(1, margin // 2))
    local_max = cv2.dilate(
        gray,
        np.ones((_odd_size(max(2, margin)), _odd_size(max(2, margin))), dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    hair = ((local_max.astype(np.int16) - gray.astype(np.int16)) >= 45) & (gray <= 135)
    hair = _dilate(hair, max(1, margin // 3))
    return CleanSkinContext(rgb, gray, valid_fov, highlight, hair, width, height, margin, outer_radius)


def build_clean_skin_mask(
    image: Image.Image,
    lesion_mask: np.ndarray,
    *,
    lesion_margin_fraction: float = 0.015,
    ring_radius_fraction: float = 0.22,
    context: CleanSkinContext | None = None,
) -> tuple[np.ndarray, dict]:
    """Return nearby skin while excluding lesion, borders, hair and highlights.

    The output is a conservative candidate mask for manual review, not a
    dermatological ground truth mask.
    """
    if context is None:
        context = prepare_clean_skin_context(
            image,
            lesion_margin_fraction=lesion_margin_fraction,
            ring_radius_fraction=ring_radius_fraction,
        )
    expected_margin = max(3, round(min(context.width, context.height) * lesion_margin_fraction))
    expected_outer = max(expected_margin + 3, round(min(context.width, context.height) * ring_radius_fraction))
    if (context.margin, context.outer_radius) != (expected_margin, expected_outer):
        raise ValueError("CleanSkinContext parameters do not match this call")
    lesion = lesion_mask.astype(bool)
    lesion_exclusion = _dilate(lesion, context.margin)
    local_ring = _dilate(lesion, context.outer_radius) & ~lesion_exclusion
    clean = local_ring & context.valid_fov & ~context.highlight & ~context.hair

    minimum_pixels = max(256, round(context.width * context.height * 0.005))
    fallback_used = False
    if int(clean.sum()) < minimum_pixels:
        clean = context.valid_fov & ~lesion_exclusion & ~context.highlight & ~context.hair
        fallback_used = True

    metadata = {
        "lesion_margin_pixels": context.margin,
        "ring_radius_pixels": context.outer_radius,
        "fallback_to_full_complement": fallback_used,
        "candidate_skin_pixels": int(clean.sum()),
    }
    return clean, metadata


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    values = rgb.astype(np.float64) / 255.0
    linear = np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)
    matrix = np.array(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]]
    )
    xyz = linear @ matrix.T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    delta = 6 / 29
    f = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3 * delta**2) + 4 / 29)
    return np.column_stack((116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])))


def measure_skin_colour(image: Image.Image, skin_mask: np.ndarray) -> SkinColorStats:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    pixels = rgb[skin_mask.astype(bool)]
    if len(pixels) == 0:
        raise ValueError("The clean-skin mask contains no pixels.")

    median = np.median(pixels, axis=0)
    lower = np.quantile(pixels, 0.10, axis=0)
    upper = np.quantile(pixels, 0.90, axis=0)
    keep = np.all((pixels >= lower) & (pixels <= upper), axis=1)
    trimmed = pixels[keep].mean(axis=0) if keep.any() else pixels.mean(axis=0)

    lab_pixels = _srgb_to_lab(pixels)
    lab = np.median(lab_pixels, axis=0)
    ita = float(np.degrees(np.arctan2(lab[0] - 50.0, lab[2])))
    return SkinColorStats(
        pixel_count=int(len(pixels)),
        coverage_fraction=float(len(pixels) / (rgb.shape[0] * rgb.shape[1])),
        rgb_median=tuple(round(float(value), 4) for value in median),
        rgb_trimmed_mean=tuple(round(float(value), 4) for value in trimmed),
        lab_median=tuple(round(float(value), 4) for value in lab),
        ita_degrees=round(ita, 4),
    )


def save_binary_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)
