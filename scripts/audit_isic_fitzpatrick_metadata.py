"""Audit a local ISIC-Fitzpatrick metadata CSV.

This script only reads an already-downloaded metadata file and writes
reproducible summary tables. It does not download images or call ISIC APIs.
"""

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "isic_id",
    "fitzpatrick_skin_type",
    "image_type",
    "diagnosis_1",
    "diagnosis_2",
    "diagnosis_3",
    "diagnosis_confirm_type",
    "copyright_license",
    "attribution",
    "patient_id",
    "lesion_id",
    "sex",
    "anatom_site_1",
]

COUNT_OUTPUTS = {
    "fitzpatrick_skin_type": "fitzpatrick_counts.csv",
    "image_type": "image_type_counts.csv",
    "diagnosis_1": "diagnosis_1_counts.csv",
    "diagnosis_2": "diagnosis_2_counts.csv",
    "diagnosis_3": "diagnosis_3_counts.csv",
    "diagnosis_confirm_type": "diagnosis_confirm_type_counts.csv",
    "copyright_license": "copyright_license_counts.csv",
    "attribution": "attribution_counts.csv",
    "sex": "sex_counts.csv",
    "anatom_site_1": "anatom_site_1_counts.csv",
}

MISSING_LABEL = "(missing)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reproducible audit summaries for ISIC-Fitzpatrick metadata."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the local metadata CSV.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where audit outputs will be written.",
    )
    return parser.parse_args()


def validate_required_columns(df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise SystemExit(f"Missing required column(s): {missing}")


def value_counts_table(series: pd.Series, total_rows: int) -> pd.DataFrame:
    counts = series.value_counts(dropna=False)
    table = counts.rename_axis("value").reset_index(name="count")
    table["value"] = table["value"].where(table["value"].notna(), MISSING_LABEL)
    table["value"] = table["value"].astype(str)
    table["percent"] = 0.0
    if total_rows > 0:
        table["percent"] = (table["count"] / total_rows * 100).round(4)

    # Stable ordering keeps regenerated outputs easy to diff.
    table = table.sort_values(["count", "value"], ascending=[False, True])
    return table[["value", "count", "percent"]]


def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        ("rows", len(df)),
        ("columns", len(df.columns)),
        ("unique_isic_id", df["isic_id"].nunique(dropna=True)),
        ("unique_patient_id", df["patient_id"].nunique(dropna=True)),
        ("unique_lesion_id", df["lesion_id"].nunique(dropna=True)),
        ("non_null_patient_id", df["patient_id"].notna().sum()),
        ("non_null_lesion_id", df["lesion_id"].notna().sum()),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value"])


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(df)
    table = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
        }
    )
    table["missing_percent"] = 0.0
    if total_rows > 0:
        table["missing_percent"] = (table["missing_count"] / total_rows * 100).round(4)
    return table.sort_values(["missing_count", "column"], ascending=[False, True])


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    display_df = df.loc[:, columns]
    if max_rows is not None:
        display_df = display_df.head(max_rows)

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in display_df.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def get_metric(overview: pd.DataFrame, metric: str) -> int:
    value = overview.loc[overview["metric"] == metric, "value"].iloc[0]
    return int(value)


def write_markdown_summary(
    output_path: Path,
    overview: pd.DataFrame,
    count_tables: dict[str, pd.DataFrame],
    missingness: pd.DataFrame,
) -> None:
    rows = get_metric(overview, "rows")
    columns = get_metric(overview, "columns")
    unique_isic_id = get_metric(overview, "unique_isic_id")
    unique_patient_id = get_metric(overview, "unique_patient_id")
    unique_lesion_id = get_metric(overview, "unique_lesion_id")
    non_null_patient_id = get_metric(overview, "non_null_patient_id")
    non_null_lesion_id = get_metric(overview, "non_null_lesion_id")

    missing_required = missingness[missingness["column"].isin(REQUIRED_COLUMNS)]

    content = [
        "# ISIC-Fitzpatrick Metadata Audit",
        "",
        "## Dataset size",
        "",
        f"- Rows: {rows}",
        f"- Columns: {columns}",
        f"- Unique `isic_id`: {unique_isic_id}",
        "",
        "## Fitzpatrick distribution",
        "",
        markdown_table(count_tables["fitzpatrick_skin_type"], ["value", "count", "percent"]),
        "",
        "## Image type distribution",
        "",
        markdown_table(count_tables["image_type"], ["value", "count", "percent"]),
        "",
        "## Diagnosis level 1",
        "",
        markdown_table(count_tables["diagnosis_1"], ["value", "count", "percent"]),
        "",
        "## Licenses",
        "",
        markdown_table(count_tables["copyright_license"], ["value", "count", "percent"]),
        "",
        "## Main sources / attributions",
        "",
        markdown_table(count_tables["attribution"], ["value", "count", "percent"], max_rows=15),
        "",
        "## Patient and lesion IDs",
        "",
        f"- Unique `patient_id`: {unique_patient_id}",
        f"- Non-null `patient_id`: {non_null_patient_id}",
        f"- Unique `lesion_id`: {unique_lesion_id}",
        f"- Non-null `lesion_id`: {non_null_lesion_id}",
        "",
        "## Required-column missingness",
        "",
        markdown_table(missing_required, ["column", "missing_count", "missing_percent"]),
        "",
        "## Brief critical interpretation",
        "",
        (
            "This metadata audit should be used before modeling to check class balance, "
            "skin-type representation, image provenance, licensing constraints, and ID "
            "coverage. Imbalances in Fitzpatrick labels, diagnoses, or acquisition sources "
            "can bias downstream evaluation. Patient and lesion identifiers are especially "
            "important for preventing leakage across train, validation, and test splits when "
            "multiple images may belong to the same person or lesion."
        ),
        "",
    ]

    output_path.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_dir = args.output_dir

    if not input_path.exists():
        raise SystemExit(f"Input CSV does not exist: {input_path}")

    df = pd.read_csv(input_path)
    validate_required_columns(df)

    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files: list[Path] = []

    overview = dataset_overview(df)
    overview_path = output_dir / "dataset_overview.csv"
    overview.to_csv(overview_path, index=False)
    generated_files.append(overview_path)

    count_tables: dict[str, pd.DataFrame] = {}
    for column, filename in COUNT_OUTPUTS.items():
        table = value_counts_table(df[column], len(df))
        count_tables[column] = table

        output_path = output_dir / filename
        table.to_csv(output_path, index=False)
        generated_files.append(output_path)

    missingness = missingness_table(df)
    missingness_path = output_dir / "missingness.csv"
    missingness.to_csv(missingness_path, index=False)
    generated_files.append(missingness_path)

    summary_path = output_dir / "isic_fitzpatrick_metadata_summary.md"
    write_markdown_summary(summary_path, overview, count_tables, missingness)
    generated_files.append(summary_path)

    print("Generated files:")
    for path in generated_files:
        print(path)


if __name__ == "__main__":
    main()
