"""Select a reproducible ISIC-Fitzpatrick pilot manifest for segmentation review.

This script builds a pilot manifest for segmentation inference and manual mask
review only. It does not create train/validation/test splits, download images,
or call ISIC APIs.
"""

import argparse
from pathlib import Path
import tomllib

import pandas as pd


KEY_COLUMNS = [
    "isic_id",
    "fitzpatrick_skin_type",
    "diagnosis_1",
    "patient_id",
    "lesion_id",
    "copyright_license",
    "attribution",
]

MANIFEST_COLUMNS = [
    "isic_id",
    "fitzpatrick_skin_type",
    "fitzpatrick_group_3_vs_3",
    "diagnosis_1",
    "binary_label",
    "image_type",
    "patient_id",
    "lesion_id",
    "copyright_license",
    "attribution",
    "age_approx",
    "sex",
    "anatom_site_1",
    "pixels_x",
    "pixels_y",
    "study_id",
]

DERIVED_COLUMNS = [
    "pilot_id",
    "selection_cell",
    "selection_reason",
    "manual_review_status",
    "mask_quality",
    "mask_notes",
]

SUMMARY_PATH = Path(
    "reports/tables/segmentation_pilot/segmentation_pilot_manifest_summary.md"
)
SELECTION_REASON = "stratified_fitzpatrick_diagnosis_pilot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a stratified pilot manifest for segmentation review."
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to study TOML.")
    parser.add_argument(
        "--per-cell",
        required=True,
        type=int,
        help="Maximum number of images to select per Fitzpatrick/diagnosis cell.",
    )
    parser.add_argument(
        "--random-state",
        required=True,
        type=int,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output CSV path for the pilot manifest.",
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


def validate_args(per_cell: int) -> None:
    if per_cell < 1:
        raise SystemExit("--per-cell must be at least 1.")


def validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise SystemExit(f"Missing required column(s): {missing}")


def known_value(value: object) -> bool:
    return not pd.isna(value)


def append_selected(
    selected_indices: list[int],
    used_patients: set[object],
    used_lesions: set[object],
    row_index: int,
    row: pd.Series,
) -> None:
    selected_indices.append(row_index)
    if known_value(row["patient_id"]):
        used_patients.add(row["patient_id"])
    if known_value(row["lesion_id"]):
        used_lesions.add(row["lesion_id"])


def select_cell(cell_df: pd.DataFrame, per_cell: int, random_state: int) -> pd.DataFrame:
    shuffled = cell_df.sample(frac=1, random_state=random_state)
    selected_indices: list[int] = []
    used_patients: set[object] = set()
    used_lesions: set[object] = set()

    # First pass: prefer one image per patient and no repeated lesion.
    for row_index, row in shuffled.iterrows():
        if len(selected_indices) >= per_cell:
            break
        patient_seen = known_value(row["patient_id"]) and row["patient_id"] in used_patients
        lesion_seen = known_value(row["lesion_id"]) and row["lesion_id"] in used_lesions
        if patient_seen or lesion_seen:
            continue
        append_selected(selected_indices, used_patients, used_lesions, row_index, row)

    # Second pass: allow repeated lesions, but still avoid repeated patients if possible.
    if len(selected_indices) < per_cell:
        for row_index, row in shuffled.iterrows():
            if len(selected_indices) >= per_cell:
                break
            if row_index in selected_indices:
                continue
            patient_seen = known_value(row["patient_id"]) and row["patient_id"] in used_patients
            if patient_seen:
                continue
            append_selected(selected_indices, used_patients, used_lesions, row_index, row)

    # Final pass: complete the cell when diversity constraints cannot fill it.
    if len(selected_indices) < per_cell:
        for row_index, row in shuffled.iterrows():
            if len(selected_indices) >= per_cell:
                break
            if row_index in selected_indices:
                continue
            append_selected(selected_indices, used_patients, used_lesions, row_index, row)

    return cell_df.loc[selected_indices].copy()


def select_manifest(
    cohort_df: pd.DataFrame,
    fitzpatrick_types: list[str],
    labels: list[str],
    per_cell: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_cells: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    cell_number = 0

    for fitzpatrick_type in fitzpatrick_types:
        for label in labels:
            cell_number += 1
            cell_mask = (
                (cohort_df["fitzpatrick_skin_type"] == fitzpatrick_type)
                & (cohort_df["diagnosis_1"] == label)
            )
            cell_df = cohort_df[cell_mask]
            available = len(cell_df)
            selected_df = select_cell(cell_df, per_cell, random_state + cell_number)
            selected = len(selected_df)

            if selected > 0:
                selected_df["selection_cell"] = f"{fitzpatrick_type}__{label}"
                selected_cells.append(selected_df)

            summary_rows.append(
                {
                    "fitzpatrick_skin_type": fitzpatrick_type,
                    "diagnosis_1": label,
                    "available": available,
                    "selected": selected,
                    "missing_vs_per_cell": max(per_cell - selected, 0),
                }
            )

    if selected_cells:
        manifest_df = pd.concat(selected_cells, ignore_index=True)
    else:
        manifest_df = pd.DataFrame(columns=cohort_df.columns.tolist() + ["selection_cell"])

    cell_summary = pd.DataFrame(summary_rows)
    return manifest_df, cell_summary


def add_manifest_columns(manifest_df: pd.DataFrame) -> pd.DataFrame:
    result = manifest_df.copy()
    result["pilot_id"] = [f"PILOT_{index:04d}" for index in range(1, len(result) + 1)]
    result["selection_reason"] = SELECTION_REASON
    result["manual_review_status"] = "pending"
    result["mask_quality"] = ""
    result["mask_notes"] = ""

    existing_manifest_columns = [
        column for column in MANIFEST_COLUMNS if column in result.columns
    ]
    return result[DERIVED_COLUMNS[:1] + existing_manifest_columns + DERIVED_COLUMNS[1:]]


def count_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    table = df[column].value_counts(dropna=False).rename_axis("value").reset_index(name="count")
    table["value"] = table["value"].where(table["value"].notna(), "(missing)")
    table["value"] = table["value"].astype(str)
    return table.sort_values(["count", "value"], ascending=[False, True])


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    display_df = df.loc[:, columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in display_df.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def write_summary(
    summary_path: Path,
    cohort_path: Path,
    output_path: Path,
    manifest_df: pd.DataFrame,
    cell_summary: pd.DataFrame,
    per_cell: int,
    random_state: int,
) -> None:
    patient_count = manifest_df["patient_id"].nunique(dropna=True)
    lesion_count = manifest_df["lesion_id"].nunique(dropna=True)

    content = [
        "# Segmentation pilot manifest summary",
        "",
        f"- Input cohort path: {cohort_path}",
        f"- Output manifest path: {output_path}",
        f"- per_cell: {per_cell}",
        f"- random_state: {random_state}",
        f"- Total selected: {len(manifest_df)}",
        f"- Unique patients: {patient_count}",
        f"- Unique lesions: {lesion_count}",
        "",
        "## Cell coverage",
        "",
        markdown_table(
            cell_summary,
            [
                "fitzpatrick_skin_type",
                "diagnosis_1",
                "available",
                "selected",
                "missing_vs_per_cell",
            ],
        ),
        "",
        "## Final Fitzpatrick distribution",
        "",
        markdown_table(count_table(manifest_df, "fitzpatrick_skin_type"), ["value", "count"]),
        "",
        "## Final diagnosis_1 distribution",
        "",
        markdown_table(count_table(manifest_df, "diagnosis_1"), ["value", "count"]),
        "",
        "## Final license distribution",
        "",
        markdown_table(count_table(manifest_df, "copyright_license"), ["value", "count"]),
        "",
        "## Top 10 attribution",
        "",
        markdown_table(count_table(manifest_df, "attribution").head(10), ["value", "count"]),
        "",
        "## Unique IDs",
        "",
        f"- Unique patients: {patient_count}",
        f"- Unique lesions: {lesion_count}",
        "",
        "## Note",
        "",
        (
            "This is not a training, validation, or test split. It is a pilot manifest "
            "for segmentation inference and manual mask review."
        ),
        "",
    ]

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args.per_cell)

    config = read_config(args.config)
    repo_root = args.config.resolve().parent.parent
    cohort_path = resolve_path(config["paths"]["cohort_output_csv"], repo_root)
    output_path = resolve_path(args.output, repo_root)
    summary_path = resolve_path(SUMMARY_PATH, repo_root)

    if not cohort_path.exists():
        raise SystemExit(f"Cohort CSV does not exist: {cohort_path}")

    cohort_df = pd.read_csv(cohort_path)
    validate_columns(cohort_df, KEY_COLUMNS)

    manifest_df, cell_summary = select_manifest(
        cohort_df=cohort_df,
        fitzpatrick_types=config["isic_query"]["fitzpatrick_types"],
        labels=config["cohort"]["include_labels"],
        per_cell=args.per_cell,
        random_state=args.random_state,
    )
    manifest_df = add_manifest_columns(manifest_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(output_path, index=False)
    write_summary(
        summary_path=summary_path,
        cohort_path=cohort_path,
        output_path=output_path,
        manifest_df=manifest_df,
        cell_summary=cell_summary,
        per_cell=args.per_cell,
        random_state=args.random_state,
    )

    print(f"Input cohort path: {cohort_path}")
    print(f"Output manifest path: {output_path}")
    print(f"Summary path: {summary_path}")
    print(f"per_cell: {args.per_cell}")
    print(f"random_state: {args.random_state}")
    print(f"Total selected: {len(manifest_df)}")
    print(f"Unique patients: {manifest_df['patient_id'].nunique(dropna=True)}")
    print(f"Unique lesions: {manifest_df['lesion_id'].nunique(dropna=True)}")


if __name__ == "__main__":
    main()
