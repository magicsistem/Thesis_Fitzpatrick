"""Create a clean-skin mask and colour summary from one lesion mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image

from _project_paths import REPO_ROOT, resolve_input, resolve_result_output

sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis_fitzpatrick.masks import (  # noqa: E402
    build_clean_skin_mask,
    measure_skin_colour,
    normalize_binary_mask,
    save_binary_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--lesion-mask", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = resolve_input(args.image)
    lesion_path = resolve_input(args.lesion_mask)
    output_dir = resolve_result_output(args.output_dir)
    if not image_path.is_file():
        raise SystemExit(f"Image does not exist: {image_path}")
    if not lesion_path.is_file():
        raise SystemExit(f"Lesion mask does not exist: {lesion_path}")

    image = Image.open(image_path).convert("RGB")
    lesion = normalize_binary_mask(Image.open(lesion_path), image.size)
    clean_skin, postprocess = build_clean_skin_mask(image, lesion)
    stats = measure_skin_colour(image, clean_skin)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_binary_mask(lesion, output_dir / "lesion_mask.png")
    save_binary_mask(clean_skin, output_dir / "clean_skin_mask.png")
    payload = {
        "image": str(image_path),
        "source_lesion_mask": str(lesion_path),
        "postprocessing": postprocess,
        "skin_colour": stats.to_dict(),
        "interpretation_warning": (
            "Fitzpatrick type is a sun-response phenotype. Colour statistics and ITA "
            "must not be treated as a replacement Fitzpatrick diagnosis."
        ),
    }
    (output_dir / "colour_stats.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(output_dir / "lesion_mask.png")
    print(output_dir / "clean_skin_mask.png")
    print(output_dir / "colour_stats.json")


if __name__ == "__main__":
    main()
