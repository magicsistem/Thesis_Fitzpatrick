"""Resize segmentation pilot images into sensor and inference versions.

The resize preserves aspect ratio, fits the full image inside the target canvas,
and pads with a constant value. Resized files are derived data and should remain
outside Git.
"""

import argparse
from pathlib import Path
import tomllib

import pandas as pd
from PIL import Image, ImageOps


REPORT_DIR = Path("reports/tables/preprocessing_resize")
RESIZE_MANIFEST = "resize_manifest.csv"
RESIZE_SUMMARY = "resize_summary.md"

REQUIRED_PREPROCESSING_KEYS = [
    "preserve_aspect_ratio",
    "padding_enabled",
    "padding_mode",
    "padding_value",
    "sensor_target_width",
    "sensor_target_height",
    "inference_target_width",
    "inference_target_height",
    "resized_sensor_image_dir",
    "resized_inference_image_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize ISIC-Fitzpatrick segmentation pilot images."
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to study TOML.")
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to segmentation pilot manifest CSV.",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing original <isic_id>.jpg files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite resized images that already exist.",
    )
    return parser.parse_args()


def read_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise SystemExit(f"Config file does not exist: {config_path}")
    with config_path.open("rb") as file:
        return tomllib.load(file)


def resolve_path(path_value: str | Path, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def validate_preprocessing_config(config: dict) -> dict:
    if "preprocessing" not in config:
        raise SystemExit("Missing [preprocessing] section in config.")

    preprocessing = config["preprocessing"]
    missing_keys = [
        key for key in REQUIRED_PREPROCESSING_KEYS if key not in preprocessing
    ]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise SystemExit(f"Missing preprocessing config key(s): {missing}")

    if preprocessing["preserve_aspect_ratio"] is not True:
        raise SystemExit("[preprocessing].preserve_aspect_ratio must be true.")
    if preprocessing["padding_enabled"] is not True:
        raise SystemExit("[preprocessing].padding_enabled must be true.")
    if preprocessing["padding_mode"] != "constant":
        raise SystemExit('[preprocessing].padding_mode must be "constant".')

    padding_value = preprocessing["padding_value"]
    if not isinstance(padding_value, int) or padding_value < 0 or padding_value > 255:
        raise SystemExit("[preprocessing].padding_value must be an integer from 0 to 255.")

    for key in [
        "sensor_target_width",
        "sensor_target_height",
        "inference_target_width",
        "inference_target_height",
    ]:
        if not isinstance(preprocessing[key], int) or preprocessing[key] < 1:
            raise SystemExit(f"[preprocessing].{key} must be a positive integer.")

    return preprocessing


def read_manifest_ids(manifest_path: Path) -> list[str]:
    if not manifest_path.exists():
        raise SystemExit(f"Manifest CSV does not exist: {manifest_path}")

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
    return ids


def validate_input_dir(input_dir: Path) -> None:
    if not input_dir.exists():
        raise SystemExit(f"Input image directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise SystemExit(f"Input path is not a directory: {input_dir}")


def resize_with_padding(
    image: Image.Image,
    target_width: int,
    target_height: int,
    padding_value: int,
) -> tuple[Image.Image, dict]:
    original_width, original_height = image.size
    width_scale = target_width / original_width
    height_scale = target_height / original_height
    scale = min(width_scale, height_scale)

    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))

    resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    resized = image.resize((resized_width, resized_height), resample=resample)

    pad_left = (target_width - resized_width) // 2
    pad_top = (target_height - resized_height) // 2
    pad_right = target_width - resized_width - pad_left
    pad_bottom = target_height - resized_height - pad_top

    canvas = Image.new(
        "RGB",
        (target_width, target_height),
        color=(padding_value, padding_value, padding_value),
    )
    canvas.paste(resized, (pad_left, pad_top))

    metadata = {
        "original_width": original_width,
        "original_height": original_height,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "pad_right": pad_right,
        "pad_bottom": pad_bottom,
    }
    return canvas, metadata


def empty_result(
    isic_id: str,
    version: str,
    input_path: Path,
    output_path: Path,
    target_width: int,
    target_height: int,
    status: str,
    error: str,
) -> dict:
    return {
        "isic_id": isic_id,
        "version": version,
        "input_path": input_path,
        "output_path": output_path,
        "original_width": pd.NA,
        "original_height": pd.NA,
        "target_width": target_width,
        "target_height": target_height,
        "resized_width": pd.NA,
        "resized_height": pd.NA,
        "pad_left": pd.NA,
        "pad_top": pd.NA,
        "pad_right": pd.NA,
        "pad_bottom": pd.NA,
        "status": status,
        "error": error,
    }


def process_version(
    isic_id: str,
    version: str,
    input_path: Path,
    output_path: Path,
    target_width: int,
    target_height: int,
    padding_value: int,
    overwrite: bool,
) -> dict:
    if not input_path.exists():
        return empty_result(
            isic_id,
            version,
            input_path,
            output_path,
            target_width,
            target_height,
            "failed",
            "Input image not found.",
        )

    try:
        with Image.open(input_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            resized_image, metadata = resize_with_padding(
                image=image,
                target_width=target_width,
                target_height=target_height,
                padding_value=padding_value,
            )

            status = "skipped" if output_path.exists() and not overwrite else "success"
            if status == "success":
                output_path.parent.mkdir(parents=True, exist_ok=True)
                resized_image.save(output_path, format="JPEG", quality=95)

            return {
                "isic_id": isic_id,
                "version": version,
                "input_path": input_path,
                "output_path": output_path,
                "original_width": metadata["original_width"],
                "original_height": metadata["original_height"],
                "target_width": target_width,
                "target_height": target_height,
                "resized_width": metadata["resized_width"],
                "resized_height": metadata["resized_height"],
                "pad_left": metadata["pad_left"],
                "pad_top": metadata["pad_top"],
                "pad_right": metadata["pad_right"],
                "pad_bottom": metadata["pad_bottom"],
                "status": status,
                "error": "",
            }
    except Exception as exc:
        return empty_result(
            isic_id,
            version,
            input_path,
            output_path,
            target_width,
            target_height,
            "failed",
            str(exc),
        )


def build_resize_manifest(
    ids: list[str],
    input_dir: Path,
    sensor_output_dir: Path,
    inference_output_dir: Path,
    preprocessing: dict,
    overwrite: bool,
) -> pd.DataFrame:
    rows: list[dict] = []
    versions = [
        (
            "sensor",
            sensor_output_dir,
            preprocessing["sensor_target_width"],
            preprocessing["sensor_target_height"],
        ),
        (
            "inference",
            inference_output_dir,
            preprocessing["inference_target_width"],
            preprocessing["inference_target_height"],
        ),
    ]

    for isic_id in ids:
        input_path = input_dir / f"{isic_id}.jpg"
        for version, output_dir, target_width, target_height in versions:
            output_path = output_dir / f"{isic_id}.jpg"
            rows.append(
                process_version(
                    isic_id=isic_id,
                    version=version,
                    input_path=input_path,
                    output_path=output_path,
                    target_width=target_width,
                    target_height=target_height,
                    padding_value=preprocessing["padding_value"],
                    overwrite=overwrite,
                )
            )

    return pd.DataFrame(rows)


def count_status(resize_manifest: pd.DataFrame, status: str) -> int:
    return int((resize_manifest["status"] == status).sum())


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    display_df = df.loc[:, columns].head(max_rows)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in display_df.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def version_status_table(resize_manifest: pd.DataFrame) -> pd.DataFrame:
    table = (
        resize_manifest.groupby(["version", "status"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    return table.sort_values(["version", "status"])


def original_size_summary(resize_manifest: pd.DataFrame) -> pd.DataFrame:
    readable = resize_manifest.dropna(subset=["original_width", "original_height"])
    readable = readable[readable["version"] == "sensor"]
    if readable.empty:
        return pd.DataFrame(columns=["original_width", "original_height", "count"])

    table = (
        readable.groupby(["original_width", "original_height"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    return table.sort_values(["count", "original_width", "original_height"], ascending=[False, True, True])


def write_summary(
    summary_path: Path,
    config_path: Path,
    manifest_path: Path,
    input_dir: Path,
    sensor_output_dir: Path,
    inference_output_dir: Path,
    total_ids: int,
    resize_manifest: pd.DataFrame,
) -> None:
    total_expected = len(resize_manifest)
    successful = count_status(resize_manifest, "success")
    skipped = count_status(resize_manifest, "skipped")
    failed = count_status(resize_manifest, "failed")

    content = [
        "# Segmentation pilot resize summary",
        "",
        f"- Config path: {config_path}",
        f"- Manifest path: {manifest_path}",
        f"- Input dir: {input_dir}",
        f"- Sensor output dir: {sensor_output_dir}",
        f"- Inference output dir: {inference_output_dir}",
        f"- Total IDs: {total_ids}",
        f"- Total expected outputs: {total_expected}",
        f"- Total successful outputs: {successful}",
        f"- Total skipped outputs: {skipped}",
        f"- Total failed outputs: {failed}",
        "",
        "## Distribution by version and status",
        "",
        markdown_table(version_status_table(resize_manifest), ["version", "status", "count"]),
        "",
        "## Original size summary",
        "",
        markdown_table(
            original_size_summary(resize_manifest),
            ["original_width", "original_height", "count"],
        ),
        "",
        "## Note",
        "",
        "Resized image files are derived artifacts and remain outside Git.",
        "",
    ]

    summary_path.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    preprocessing = validate_preprocessing_config(config)

    repo_root = args.config.resolve().parent.parent
    manifest_path = resolve_path(args.manifest, repo_root)
    input_dir = resolve_path(args.input_dir, repo_root)
    sensor_output_dir = resolve_path(preprocessing["resized_sensor_image_dir"], repo_root)
    inference_output_dir = resolve_path(preprocessing["resized_inference_image_dir"], repo_root)
    report_dir = repo_root / REPORT_DIR

    validate_input_dir(input_dir)
    ids = read_manifest_ids(manifest_path)

    sensor_output_dir.mkdir(parents=True, exist_ok=True)
    inference_output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    resize_manifest = build_resize_manifest(
        ids=ids,
        input_dir=input_dir,
        sensor_output_dir=sensor_output_dir,
        inference_output_dir=inference_output_dir,
        preprocessing=preprocessing,
        overwrite=args.overwrite,
    )

    resize_manifest_path = report_dir / RESIZE_MANIFEST
    summary_path = report_dir / RESIZE_SUMMARY
    resize_manifest.to_csv(resize_manifest_path, index=False)
    write_summary(
        summary_path=summary_path,
        config_path=args.config,
        manifest_path=manifest_path,
        input_dir=input_dir,
        sensor_output_dir=sensor_output_dir,
        inference_output_dir=inference_output_dir,
        total_ids=len(ids),
        resize_manifest=resize_manifest,
    )

    total_expected = len(resize_manifest)
    successful = count_status(resize_manifest, "success")
    skipped = count_status(resize_manifest, "skipped")
    failed = count_status(resize_manifest, "failed")

    print(f"Total IDs: {len(ids)}")
    print(f"Total expected outputs: {total_expected}")
    print(f"Successful: {successful}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Output report path: {resize_manifest_path}")


if __name__ == "__main__":
    main()
