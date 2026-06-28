"""Download images listed in the ISIC-Fitzpatrick segmentation pilot manifest.

This script uses the installed `isic` CLI. It does not call the ISIC HTTP API
directly, install packages, or create notebooks.
"""

import argparse
import math
from pathlib import Path
import shlex
import subprocess
import time

import pandas as pd


SUMMARY_DIR = Path("reports/tables/segmentation_pilot_download")
DOWNLOAD_BATCHES_CSV = "download_batches.csv"
DOWNLOAD_SUMMARY_MD = "download_summary.md"
DOWNLOADED_IDS_CSV = "downloaded_ids.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download only images listed in a segmentation pilot manifest."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to the segmentation pilot manifest CSV.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where the ISIC CLI will write images and metadata.",
    )
    parser.add_argument(
        "--batch-size",
        required=True,
        type=int,
        help="Number of ISIC IDs per download batch.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear the output directory before downloading.",
    )
    parser.add_argument(
        "--sleep-seconds",
        default=0.0,
        type=float,
        help="Seconds to sleep between download batches.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.manifest.exists():
        raise SystemExit(f"Manifest CSV does not exist: {args.manifest}")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")
    if args.sleep_seconds < 0:
        raise SystemExit("--sleep-seconds must be 0 or greater.")


def ensure_isic_cli_available() -> None:
    try:
        subprocess.run(
            ["isic", "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            "The `isic` executable is not available. Activate the `isic-api` "
            "environment before running this script."
        ) from exc


def read_manifest_ids(manifest_path: Path) -> list[str]:
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


def split_batches(ids: list[str], batch_size: int) -> list[list[str]]:
    total_batches = math.ceil(len(ids) / batch_size) if ids else 0
    return [
        ids[index * batch_size : (index + 1) * batch_size]
        for index in range(total_batches)
    ]


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return

    if path.is_dir():
        for child in path.iterdir():
            remove_path(child)
        path.rmdir()


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if overwrite and output_dir.exists():
        for child in output_dir.iterdir():
            remove_path(child)
    output_dir.mkdir(parents=True, exist_ok=True)


def build_search_query(ids: list[str]) -> str:
    return " OR ".join(f"isic_id:{isic_id}" for isic_id in ids)


def image_path(output_dir: Path, isic_id: str) -> Path:
    return output_dir / f"{isic_id}.jpg"


def present_manifest_ids(output_dir: Path, manifest_ids: list[str]) -> list[str]:
    return [isic_id for isic_id in manifest_ids if image_path(output_dir, isic_id).exists()]


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def run_batch(
    batch_ids: list[str],
    batch_index: int,
    output_dir: Path,
    summary_dir: Path,
    overwrite: bool,
) -> dict:
    stdout_path = summary_dir / f"batch_{batch_index:03d}.stdout.txt"
    stderr_path = summary_dir / f"batch_{batch_index:03d}.stderr.txt"
    query = build_search_query(batch_ids)
    command = [
        "isic",
        "image",
        "download",
        "--search",
        query,
        "--limit",
        "0",
        str(output_dir),
    ]

    all_present = all(image_path(output_dir, isic_id).exists() for isic_id in batch_ids)
    if all_present and not overwrite:
        write_text(stdout_path, "Skipped: all requested images are already present.\n")
        write_text(stderr_path, "")
        returncode = 0
        status = "skipped_present"
    else:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        write_text(
            stdout_path,
            f"$ {shlex.join(command)}\n\n{completed.stdout}",
        )
        write_text(stderr_path, completed.stderr)
        returncode = completed.returncode
        status = "success" if returncode == 0 else "failed"

    return {
        "batch_index": batch_index,
        "n_requested": len(batch_ids),
        "first_isic_id": batch_ids[0] if batch_ids else "",
        "last_isic_id": batch_ids[-1] if batch_ids else "",
        "query_length": len(query),
        "returncode": returncode,
        "status": status,
        "stdout_log_path": stdout_path,
        "stderr_log_path": stderr_path,
    }


def count_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    table = df[column].value_counts(dropna=False).rename_axis("value").reset_index(name="count")
    table["value"] = table["value"].where(table["value"].notna(), "(missing)")
    table["value"] = table["value"].astype(str)
    return table.sort_values(["count", "value"], ascending=[False, True])


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    display_df = df.loc[:, columns].head(max_rows)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in display_df.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def write_downloaded_ids(
    summary_dir: Path,
    output_dir: Path,
    manifest_ids: list[str],
) -> pd.DataFrame:
    downloaded_ids = present_manifest_ids(output_dir, manifest_ids)
    downloaded_df = pd.DataFrame({"isic_id": downloaded_ids})
    downloaded_df.to_csv(summary_dir / DOWNLOADED_IDS_CSV, index=False)
    return downloaded_df


def write_summary(
    summary_path: Path,
    manifest_path: Path,
    output_dir: Path,
    batch_size: int,
    total_ids: int,
    total_batches: int,
    downloaded_count: int,
    missing_count: int,
    batch_df: pd.DataFrame,
) -> None:
    content = [
        "# Segmentation pilot image download summary",
        "",
        f"- Manifest path: {manifest_path}",
        f"- Output dir: {output_dir}",
        f"- Batch size: {batch_size}",
        f"- Total IDs in manifest: {total_ids}",
        f"- Total batches: {total_batches}",
        f"- Total downloaded images detected on disk: {downloaded_count}",
        f"- Total missing relative to manifest: {missing_count}",
        "",
        "## Batch status",
        "",
        markdown_table(
            batch_df,
            [
                "batch_index",
                "n_requested",
                "first_isic_id",
                "last_isic_id",
                "returncode",
                "status",
            ],
        ),
        "",
        "## Status counts",
        "",
        markdown_table(count_table(batch_df, "status"), ["value", "count"]),
        "",
        "## Note",
        "",
        "Downloaded ISIC images and CLI-generated metadata remain outside Git.",
        "",
    ]
    write_text(summary_path, "\n".join(content))


def main() -> None:
    args = parse_args()
    validate_args(args)
    ensure_isic_cli_available()

    repo_root = Path(__file__).resolve().parent.parent
    summary_dir = repo_root / SUMMARY_DIR
    summary_dir.mkdir(parents=True, exist_ok=True)

    manifest_ids = read_manifest_ids(args.manifest)
    batches = split_batches(manifest_ids, args.batch_size)

    prepare_output_dir(args.output_dir, args.overwrite)

    batch_rows = []
    for batch_index, batch_ids in enumerate(batches, start=1):
        batch_rows.append(
            run_batch(
                batch_ids=batch_ids,
                batch_index=batch_index,
                output_dir=args.output_dir,
                summary_dir=summary_dir,
                overwrite=args.overwrite,
            )
        )
        if args.sleep_seconds > 0 and batch_index < len(batches):
            time.sleep(args.sleep_seconds)

    batch_df = pd.DataFrame(batch_rows)
    batch_df.to_csv(summary_dir / DOWNLOAD_BATCHES_CSV, index=False)

    downloaded_df = write_downloaded_ids(summary_dir, args.output_dir, manifest_ids)
    downloaded_count = len(downloaded_df)
    missing_count = len(manifest_ids) - downloaded_count

    summary_path = summary_dir / DOWNLOAD_SUMMARY_MD
    write_summary(
        summary_path=summary_path,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        total_ids=len(manifest_ids),
        total_batches=len(batches),
        downloaded_count=downloaded_count,
        missing_count=missing_count,
        batch_df=batch_df,
    )

    print(f"Manifest path: {args.manifest}")
    print(f"Output dir: {args.output_dir}")
    print(f"Total IDs: {len(manifest_ids)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Total batches: {len(batches)}")
    print(f"Downloaded image count detected on disk: {downloaded_count}")
    print(f"Missing image count: {missing_count}")


if __name__ == "__main__":
    main()
