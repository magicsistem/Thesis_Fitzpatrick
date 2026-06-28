"""Run a small MobileSAM prompt pilot for visual mask inspection.

This is a pilot script for a few resized ISIC-Fitzpatrick images. It uses one
positive center point and four negative corner points, then saves all three
MobileSAM mask candidates plus red overlays for manual review.
"""

import argparse
from pathlib import Path
import time

from mobile_sam import SamPredictor, sam_model_registry
import numpy as np
import pandas as pd
from PIL import Image
import torch


RESULTS_DIR = Path("results/mobile_sam_prompt_pilot")
SUMMARY_CSV = "pilot_summary.csv"
SUMMARY_MD = "pilot_summary.md"
MANUAL_REVIEW_CSV = "manual_review_template.csv"

MANUAL_REVIEW_COLUMNS = [
    "isic_id",
    "reviewer",
    "best_mask_index",
    "mask_0_quality",
    "mask_1_quality",
    "mask_2_quality",
    "overall_usable",
    "failure_low_lesion_skin_contrast",
    "failure_irregular_or_dispersed_lesion",
    "failure_multiple_lesions",
    "failure_dermatoscope_tip_artifact",
    "failure_background_or_skin_selected",
    "failure_other",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small MobileSAM prompt pilot with simple point prompts."
    )
    parser.add_argument("--manifest", required=True, type=Path, help="Pilot manifest CSV.")
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing resized inference <isic_id>.jpg images.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Local MobileSAM checkpoint path.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where per-image masks and overlays will be written.",
    )
    parser.add_argument(
        "--limit",
        required=True,
        type=int,
        help="Number of manifest images to process.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.manifest.exists():
        raise SystemExit(f"Manifest CSV does not exist: {args.manifest}")
    if not args.input_dir.exists():
        raise SystemExit(f"Input image directory does not exist: {args.input_dir}")
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input path is not a directory: {args.input_dir}")
    if not args.checkpoint.exists():
        raise SystemExit(f"MobileSAM checkpoint does not exist: {args.checkpoint}")
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1.")


def read_limited_ids(manifest_path: Path, limit: int) -> list[str]:
    manifest = pd.read_csv(manifest_path)
    if "isic_id" not in manifest.columns:
        raise SystemExit("Manifest CSV is missing required column: isic_id")

    ids: list[str] = []
    seen: set[str] = set()
    for value in manifest["isic_id"].dropna():
        isic_id = str(value).strip()
        if isic_id and isic_id not in seen:
            ids.append(isic_id)
            seen.add(isic_id)
        if len(ids) >= limit:
            break
    return ids


def load_predictor(checkpoint_path: Path) -> SamPredictor:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = sam_model_registry["vit_t"](checkpoint=str(checkpoint_path))
    model.to(device=device)
    model.eval()
    return SamPredictor(model)


def prompt_points(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    center_x = width / 2.0
    center_y = height / 2.0
    points = np.array(
        [
            [center_x, center_y],
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [0.0, height - 1.0],
            [width - 1.0, height - 1.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([1, 0, 0, 0, 0], dtype=np.int32)
    return points, labels


def save_mask(mask: np.ndarray, output_path: Path) -> int:
    mask_uint8 = (mask.astype(np.uint8) * 255)
    Image.fromarray(mask_uint8, mode="L").save(output_path)
    return int(mask.sum())


def save_overlay(image: Image.Image, mask: np.ndarray, output_path: Path) -> None:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    alpha = (mask.astype(np.uint8) * 110)
    red = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    red[:, :, 0] = 255
    red[:, :, 3] = alpha
    overlay = Image.fromarray(red, mode="RGBA")
    blended = Image.alpha_composite(base, overlay)
    blended.save(output_path)


def success_row(
    isic_id: str,
    mask_index: int,
    score: float,
    area: int,
    width: int,
    height: int,
    mask_path: Path,
    overlay_path: Path,
    inference_seconds: float,
) -> dict:
    return {
        "isic_id": isic_id,
        "mask_index": mask_index,
        "score": score,
        "area": area,
        "area_fraction": area / (width * height),
        "image_width": width,
        "image_height": height,
        "mask_path": mask_path,
        "overlay_path": overlay_path,
        "status": "success",
        "error": "",
        "inference_seconds": round(inference_seconds, 4),
    }


def error_row(isic_id: str, error: str, inference_seconds: float) -> dict:
    return {
        "isic_id": isic_id,
        "mask_index": -1,
        "score": np.nan,
        "area": np.nan,
        "area_fraction": np.nan,
        "image_width": np.nan,
        "image_height": np.nan,
        "mask_path": "",
        "overlay_path": "",
        "status": "failed",
        "error": error,
        "inference_seconds": round(inference_seconds, 4),
    }


def process_image(
    predictor: SamPredictor,
    isic_id: str,
    input_dir: Path,
    output_dir: Path,
) -> list[dict]:
    start = time.perf_counter()
    image_path = input_dir / f"{isic_id}.jpg"
    image_output_dir = output_dir / isic_id
    image_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        height, width = image_array.shape[:2]
        point_coords, point_labels = prompt_points(width, height)

        predictor.set_image(image_array)
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

        inference_seconds = time.perf_counter() - start
        rows = []
        for mask_index, mask in enumerate(masks):
            mask_path = image_output_dir / f"mask_{mask_index}.png"
            overlay_path = image_output_dir / f"overlay_{mask_index}.png"
            area = save_mask(mask, mask_path)
            save_overlay(image, mask, overlay_path)
            rows.append(
                success_row(
                    isic_id=isic_id,
                    mask_index=mask_index,
                    score=float(scores[mask_index]),
                    area=area,
                    width=width,
                    height=height,
                    mask_path=mask_path,
                    overlay_path=overlay_path,
                    inference_seconds=inference_seconds,
                )
            )
        return rows
    except Exception as exc:
        inference_seconds = time.perf_counter() - start
        return [error_row(isic_id, str(exc), inference_seconds)]


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    display_df = df.loc[:, columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in display_df.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def mask_index_counts(summary_df: pd.DataFrame) -> pd.DataFrame:
    successful = summary_df[summary_df["status"] == "success"]
    if successful.empty:
        return pd.DataFrame(columns=["mask_index", "count"])
    table = successful["mask_index"].value_counts().rename_axis("mask_index").reset_index(name="count")
    return table.sort_values("mask_index")


def write_manual_review_template(isic_ids: list[str], output_path: Path) -> None:
    rows = []
    for isic_id in isic_ids:
        row = {column: "" for column in MANUAL_REVIEW_COLUMNS}
        row["isic_id"] = isic_id
        rows.append(row)

    pd.DataFrame(rows, columns=MANUAL_REVIEW_COLUMNS).to_csv(output_path, index=False)


def write_markdown_summary(
    summary_path: Path,
    summary_df: pd.DataFrame,
    total_requested: int,
    total_seconds: float,
) -> None:
    failed_images = summary_df.loc[summary_df["status"] == "failed", "isic_id"].nunique()
    processed_images = summary_df.loc[summary_df["status"] == "success", "isic_id"].nunique()
    counts = mask_index_counts(summary_df)

    content = [
        "# MobileSAM prompt pilot summary",
        "",
        f"- Total images requested: {total_requested}",
        f"- Total processed: {processed_images}",
        f"- Total with error: {failed_images}",
        f"- Total time seconds: {round(total_seconds, 4)}",
        "",
        "## Counts by mask_index",
        "",
        markdown_table(counts, ["mask_index", "count"]),
        "",
        "## Note",
        "",
        (
            "This is a pilot for visual inspection of prompt-based MobileSAM masks, "
            "not the final batch inference run."
        ),
        "",
        "## Methodological note",
        "",
        (
            "The masks generated by MobileSAM in this pilot must not be treated as "
            "ground truth. They should not be used for classification without manual "
            "review and quality control. Preliminary visual review showed important "
            "failure modes, including low lesion-skin contrast, irregular or dispersed "
            "lesions, multiple lesions, dermatoscope-tip artifacts, and selection of "
            "background or surrounding skin instead of the lesion."
        ),
        "",
    ]
    summary_path.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)

    repo_root = Path(__file__).resolve().parent.parent
    results_dir = repo_root / RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    isic_ids = read_limited_ids(args.manifest, args.limit)
    predictor = load_predictor(args.checkpoint)

    all_rows: list[dict] = []
    total_start = time.perf_counter()
    for index, isic_id in enumerate(isic_ids, start=1):
        print(f"[{index}/{len(isic_ids)}] Processing {isic_id}")
        rows = process_image(
            predictor=predictor,
            isic_id=isic_id,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
        )
        all_rows.extend(rows)
        status = "ok" if rows and rows[0]["status"] == "success" else "failed"
        print(f"[{index}/{len(isic_ids)}] {isic_id}: {status}")

    total_seconds = time.perf_counter() - total_start
    summary_df = pd.DataFrame(all_rows)

    summary_csv_path = results_dir / SUMMARY_CSV
    summary_md_path = results_dir / SUMMARY_MD
    manual_review_path = results_dir / MANUAL_REVIEW_CSV
    summary_df.to_csv(summary_csv_path, index=False)
    write_manual_review_template(isic_ids, manual_review_path)
    write_markdown_summary(
        summary_path=summary_md_path,
        summary_df=summary_df,
        total_requested=len(isic_ids),
        total_seconds=total_seconds,
    )

    failed_images = summary_df.loc[summary_df["status"] == "failed", "isic_id"].nunique()
    processed_images = summary_df.loc[summary_df["status"] == "success", "isic_id"].nunique()
    print(f"Total images requested: {len(isic_ids)}")
    print(f"Total processed: {processed_images}")
    print(f"Total with error: {failed_images}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Summary Markdown: {summary_md_path}")
    print(f"Manual review template: {manual_review_path}")


if __name__ == "__main__":
    main()
