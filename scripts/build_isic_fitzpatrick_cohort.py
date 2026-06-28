"""Build the first ISIC-Fitzpatrick experimental cohort from local metadata.

This script reads only local files defined in the study TOML configuration.
It does not download images or call ISIC APIs.
"""

import argparse
from pathlib import Path
import tomllib

import pandas as pd


LIGHTER_GROUP = "lighter_I_III"
DARKER_GROUP = "darker_IV_VI"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the configured ISIC-Fitzpatrick cohort."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the TOML study configuration.",
    )
    return parser.parse_args()


def read_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise SystemExit(f"Config file does not exist: {config_path}")
    with config_path.open("rb") as file:
        return tomllib.load(file)


def resolve_path(path_value: str, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise SystemExit(f"Missing required column(s): {missing}")


def apply_filters(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    cohort = config["cohort"]
    exclusions = config["exclusions"]
    label_column = cohort["label_column"]

    filtered = df.copy()

    if exclusions.get("exclude_non_dermoscopic", False):
        filtered = filtered[filtered["image_type"].isin(cohort["include_image_type"])]
    else:
        filtered = filtered[filtered["image_type"].isin(cohort["include_image_type"])]

    if exclusions.get("exclude_missing_label", False):
        filtered = filtered[filtered[label_column].notna()]

    filtered = filtered[filtered[label_column].isin(cohort["include_labels"])]
    filtered = filtered[~filtered[label_column].isin(cohort["exclude_labels"])]

    if exclusions.get("exclude_missing_fitzpatrick", False):
        filtered = filtered[filtered["fitzpatrick_skin_type"].notna()]

    if exclusions.get("exclude_missing_patient_id", False):
        filtered = filtered[filtered["patient_id"].notna()]

    return filtered.copy()


def add_derived_columns(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    cohort = config["cohort"]
    fitzpatrick = config["fitzpatrick_analysis"]
    label_column = cohort["label_column"]

    result = df.copy()
    label_map = {
        cohort["negative_label"]: 0,
        cohort["positive_label"]: 1,
    }
    result["binary_label"] = result[label_column].map(label_map)

    lighter_types = set(fitzpatrick["lighter_skin_group"])
    darker_types = set(fitzpatrick["darker_skin_group"])
    result["fitzpatrick_group_3_vs_3"] = result["fitzpatrick_skin_type"].map(
        lambda value: LIGHTER_GROUP
        if value in lighter_types
        else DARKER_GROUP
        if value in darker_types
        else pd.NA
    )

    result["study_id"] = config["project"]["study_id"]
    return result


def validate_values(df: pd.DataFrame, config: dict) -> None:
    expected_fitzpatrick = set(config["isic_query"]["fitzpatrick_types"])
    observed_fitzpatrick = set(df["fitzpatrick_skin_type"].dropna().unique())
    unexpected_fitzpatrick = sorted(observed_fitzpatrick - expected_fitzpatrick)
    if unexpected_fitzpatrick:
        values = ", ".join(str(value) for value in unexpected_fitzpatrick)
        raise SystemExit(f"Unexpected fitzpatrick_skin_type value(s): {values}")

    label_column = config["cohort"]["label_column"]
    expected_labels = set(config["cohort"]["include_labels"])
    observed_labels = set(df[label_column].dropna().unique())
    unexpected_labels = sorted(observed_labels - expected_labels)
    if unexpected_labels:
        values = ", ".join(str(value) for value in unexpected_labels)
        raise SystemExit(f"Unexpected {label_column} value(s): {values}")

    observed_binary = set(df["binary_label"].dropna().unique())
    unexpected_binary = sorted(observed_binary - {0, 1})
    if unexpected_binary:
        values = ", ".join(str(value) for value in unexpected_binary)
        raise SystemExit(f"Unexpected binary_label value(s): {values}")

    if df["binary_label"].isna().any():
        raise SystemExit("binary_label contains missing values after cohort filtering.")

    if df["fitzpatrick_group_3_vs_3"].isna().any():
        raise SystemExit(
            "fitzpatrick_group_3_vs_3 contains missing values after cohort filtering."
        )


def count_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    table = df[column].value_counts(dropna=False).rename_axis("value").reset_index(name="count")
    table["value"] = table["value"].where(table["value"].notna(), "(missing)")
    table["value"] = table["value"].astype(str)
    table = table.sort_values(["count", "value"], ascending=[False, True])
    return table[["value", "count"]]


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


def write_summary(
    summary_path: Path,
    df: pd.DataFrame,
    rows_before: int,
    config: dict,
) -> None:
    rows_after = len(df)
    patient_count = df["patient_id"].nunique(dropna=True)
    lesion_count = df["lesion_id"].nunique(dropna=True)

    tables = {
        "diagnosis_1": count_table(df, "diagnosis_1"),
        "binary_label": count_table(df, "binary_label"),
        "fitzpatrick_skin_type": count_table(df, "fitzpatrick_skin_type"),
        "fitzpatrick_group_3_vs_3": count_table(df, "fitzpatrick_group_3_vs_3"),
        "image_type": count_table(df, "image_type"),
        "copyright_license": count_table(df, "copyright_license"),
        "attribution": count_table(df, "attribution"),
    }

    content = [
        f"# {config['project']['study_id']} cohort summary",
        "",
        "## Size",
        "",
        f"- Rows before filtering: {rows_before}",
        f"- Rows after filtering: {rows_after}",
        f"- Unique patients: {patient_count}",
        f"- Unique lesions: {lesion_count}",
        "",
        "## diagnosis_1 distribution",
        "",
        markdown_table(tables["diagnosis_1"], ["value", "count"]),
        "",
        "## binary_label distribution",
        "",
        markdown_table(tables["binary_label"], ["value", "count"]),
        "",
        "## fitzpatrick_skin_type distribution",
        "",
        markdown_table(tables["fitzpatrick_skin_type"], ["value", "count"]),
        "",
        "## fitzpatrick_group_3_vs_3 distribution",
        "",
        markdown_table(tables["fitzpatrick_group_3_vs_3"], ["value", "count"]),
        "",
        "## image_type distribution",
        "",
        markdown_table(tables["image_type"], ["value", "count"]),
        "",
        "## copyright_license distribution",
        "",
        markdown_table(tables["copyright_license"], ["value", "count"]),
        "",
        "## Top 10 attribution",
        "",
        markdown_table(tables["attribution"], ["value", "count"], max_rows=10),
        "",
        "## Leakage note",
        "",
        "Splits for modeling must be created by `patient_id` to reduce leakage risk.",
        "",
    ]

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = args.config
    repo_root = config_path.resolve().parent.parent
    config = read_config(config_path)

    raw_metadata_path = resolve_path(config["paths"]["raw_metadata_csv"], repo_root)
    cohort_output_path = resolve_path(config["paths"]["cohort_output_csv"], repo_root)
    summary_path = repo_root / "reports/tables/cohort_build/isic_fitzpatrick_dermoscopic_binary_v1_summary.md"

    if not raw_metadata_path.exists():
        raise SystemExit(f"Raw metadata CSV does not exist: {raw_metadata_path}")

    df = pd.read_csv(raw_metadata_path)
    validate_columns(df, config["cohort"]["required_columns"])

    rows_before = len(df)
    cohort_df = apply_filters(df, config)
    cohort_df = add_derived_columns(cohort_df, config)
    validate_values(cohort_df, config)

    cohort_output_path.parent.mkdir(parents=True, exist_ok=True)
    cohort_df.to_csv(cohort_output_path, index=False)

    write_summary(summary_path, cohort_df, rows_before, config)

    print(f"Input path: {raw_metadata_path}")
    print(f"Output cohort path: {cohort_output_path}")
    print(f"Summary path: {summary_path}")
    print(f"Rows before filtering: {rows_before}")
    print(f"Rows after filtering: {len(cohort_df)}")
    print(f"Unique patients: {cohort_df['patient_id'].nunique(dropna=True)}")
    print(f"Unique lesions: {cohort_df['lesion_id'].nunique(dropna=True)}")


if __name__ == "__main__":
    main()
