"""Aggregate completed runs and perform paired statistical comparisons."""

from __future__ import annotations

import csv
from itertools import combinations
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from .benchmark import atomic_write_bytes, atomic_write_json, is_fatal_backend_failure, load_json
from .metrics import holm_adjust, mcnemar_exact, paired_bootstrap, paired_permutation_pvalue


METRICS = (
    "threshold_jaccard", "jaccard", "dice", "sensitivity", "specificity", "precision",
    "accuracy", "boundary_f1", "hd95_pixels", "hd95_normalized", "fov_leak", "clean_skin_contamination",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    lines: list[str] = []
    if columns:
        import io
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        lines.append(stream.getvalue())
    atomic_write_bytes(path, "".join(lines).encode())


def load_run_rows(run_directory: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((run_directory / "predictions").rglob("result.json")):
        payload = load_json(path)
        metrics = payload.get("metrics") or {}
        metadata = payload.get("metadata") or {}
        flags = metrics.get("flags") or {}
        failure_code = payload.get("failure_code") or (payload.get("backend") or {}).get("failure_code")
        technical_failure = is_fatal_backend_failure(failure_code)
        prediction_empty = bool(flags.get("prediction_empty"))
        prediction_nearly_complete = bool(flags.get("prediction_nearly_complete"))
        rows.append({
            "image_id": payload["image_id"],
            "method_id": payload["method_id"],
            "condition": payload.get("condition") or payload.get("evaluation"),
            "fitzpatrick": metadata.get("fitzpatrick"),
            "failure_code": failure_code,
            "failure": technical_failure,
            "technical_failure": technical_failure,
            "prediction_empty": prediction_empty,
            "prediction_nearly_complete": prediction_nearly_complete,
            "degenerate_prediction": (not technical_failure) and bool(failure_code or prediction_empty or prediction_nearly_complete),
            "fallback": bool(payload.get("fallback_used") or payload.get("p0_fallback_used")),
            "backend_time_ms": (payload.get("backend") or {}).get("backend_time_ms"),
            "end_to_end_time_ms": payload.get("end_to_end_time_ms"),
            **{key: metrics.get(key) for key in METRICS},
        })
    return rows


def _ci(values: np.ndarray, repetitions: int, seed: int) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    samples = values[generator.integers(0, len(values), size=(repetitions, len(values)))].mean(axis=1)
    return tuple(float(value) for value in np.percentile(samples, (2.5, 97.5)))


def aggregate(rows: list[dict[str, Any]], *, repetitions: int = 10_000, seed: int = 20260806) -> list[dict[str, Any]]:
    summaries = []
    keys = sorted({(row["condition"], row["method_id"]) for row in rows})
    for condition, method_id in keys:
        subset = [row for row in rows if (row["condition"], row["method_id"]) == (condition, method_id)]
        scientific_subset = [row for row in subset if not row["technical_failure"]]
        result: dict[str, Any] = {
            "condition": condition, "method_id": method_id, "n": len(subset),
            "scientific_n": len(scientific_subset),
            "failures": sum(row["technical_failure"] for row in subset),
            "technical_failures": sum(row["technical_failure"] for row in subset),
            "degenerate_predictions": sum(row["degenerate_prediction"] for row in subset),
            "prediction_empty": sum(row["prediction_empty"] for row in subset),
            "prediction_nearly_complete": sum(row["prediction_nearly_complete"] for row in subset),
            "fallbacks": sum(row["fallback"] for row in subset),
        }
        for metric in METRICS:
            values = np.asarray([row[metric] for row in scientific_subset if row.get(metric) is not None and math.isfinite(float(row[metric]))], dtype=float)
            if len(values):
                low, high = _ci(values, repetitions, seed)
                result.update({f"{metric}_mean": float(mean(values)), f"{metric}_median": float(median(values)), f"{metric}_ci95_low": low, f"{metric}_ci95_high": high})
        group_values: dict[str, float] = {}
        for group in sorted({str(row["fitzpatrick"]) for row in subset if row.get("fitzpatrick") not in (None, "")}):
            values = [float(row["threshold_jaccard"]) for row in subset if str(row.get("fitzpatrick")) == group and row.get("threshold_jaccard") is not None]
            if values:
                group_values[group] = mean(values)
        result["fitzpatrick_group_threshold_jaccard"] = group_values
        result["fitzpatrick_worst_group"] = min(group_values.values()) if group_values else None
        result["fitzpatrick_best_minus_worst"] = max(group_values.values()) - min(group_values.values()) if group_values else None
        summaries.append(result)
    return summaries


def paired_comparisons(rows: list[dict[str, Any]], *, repetitions: int = 10_000, seed: int = 20260806) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for condition in sorted({row["condition"] for row in rows}):
        subset = [row for row in rows if row["condition"] == condition]
        by_method = {
            method: {
                row["image_id"]: row
                for row in subset
                if row["method_id"] == method and not row["technical_failure"]
            }
            for method in sorted({row["method_id"] for row in subset})
        }
        for first, second in combinations(by_method, 2):
            image_ids = sorted(set(by_method[first]) & set(by_method[second]))
            for metric in ("threshold_jaccard", "jaccard", "dice", "boundary_f1", "hd95_normalized", "clean_skin_contamination"):
                pairs = [(by_method[first][key].get(metric), by_method[second][key].get(metric)) for key in image_ids]
                pairs = [(a, b) for a, b in pairs if a is not None and b is not None and math.isfinite(float(a)) and math.isfinite(float(b))]
                if not pairs:
                    continue
                one, two = (np.asarray(values, dtype=float) for values in zip(*pairs))
                boot = paired_bootstrap(one, two, repetitions=repetitions, seed=seed)
                comparisons.append({"condition": condition, "first": first, "second": second, "metric": metric, "n": len(one), **boot, "permutation_pvalue": paired_permutation_pvalue(one, two, repetitions=repetitions, seed=seed)})
            success_one = np.asarray([bool((by_method[first][key].get("jaccard") or 0) >= 0.65) for key in image_ids])
            success_two = np.asarray([bool((by_method[second][key].get("jaccard") or 0) >= 0.65) for key in image_ids])
            if len(image_ids):
                comparisons.append({"condition": condition, "first": first, "second": second, "metric": "success_jaccard_0.65", "n": len(image_ids), **mcnemar_exact(success_one, success_two), "permutation_pvalue": mcnemar_exact(success_one, success_two)["pvalue"]})
    adjusted = holm_adjust([float(row["permutation_pvalue"]) for row in comparisons]) if comparisons else []
    for row, value in zip(comparisons, adjusted):
        row["holm_adjusted_pvalue"] = value
    return comparisons


def write_report(run_directory: Path, *, repetitions: int = 10_000, seed: int = 20260806) -> dict[str, Any]:
    rows = load_run_rows(run_directory)
    summaries = aggregate(rows, repetitions=repetitions, seed=seed)
    comparisons = paired_comparisons(rows, repetitions=repetitions, seed=seed)
    _write_csv(run_directory / "metrics_summary.csv", summaries)
    _write_csv(run_directory / "statistical_comparisons.csv", comparisons)
    lines = ["| Condición | Método | N | N científico | Fallos técnicos | Degeneradas | Vacías | TJ media | TJ mediana | IC95% |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in summaries:
        lines.append(f"| {row['condition']} | {row['method_id']} | {row['n']} | {row['scientific_n']} | {row['technical_failures']} | {row['degenerate_predictions']} | {row['prediction_empty']} | {row.get('threshold_jaccard_mean', float('nan')):.4f} | {row.get('threshold_jaccard_median', float('nan')):.4f} | [{row.get('threshold_jaccard_ci95_low', float('nan')):.4f}, {row.get('threshold_jaccard_ci95_high', float('nan')):.4f}] |")
    atomic_write_bytes(run_directory / "metrics_summary.md", ("\n".join(lines) + "\n").encode())
    report = {
        "schema_version": 1,
        "run_id": run_directory.name,
        "rows": len(rows),
        "scientific_rows": sum(not row["technical_failure"] for row in rows),
        "outcome_convention": "Degenerate predictions remain scientific observations; technical failures are excluded until rerun.",
        "summaries": summaries,
        "pairwise_comparisons": comparisons,
        "bootstrap_repetitions": repetitions,
        "seed": seed,
    }
    atomic_write_json(run_directory / "report.json", report)
    return report
