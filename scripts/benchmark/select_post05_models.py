#!/usr/bin/env python3
"""Rank TOP-3 segmenters and freeze all scientific decisions before MSKCC."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis_fitzpatrick.benchmark import atomic_write_bytes, atomic_write_json, content_hash, load_json, sha256_file  # noqa: E402
from thesis_fitzpatrick.post05 import choose_top3, validate_final_yolo_manifest  # noqa: E402
from thesis_fitzpatrick.reporting import load_run_rows  # noqa: E402


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=False, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, check=False, capture_output=True, text=True).stdout.strip())
    return {"commit": commit or None, "dirty": dirty}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat: list[dict[str, Any]] = []
    for row in rows:
        flat.append({
            "rank": row.get("rank"),
            "method_id": row["method_id"],
            "eligible": row["eligible"],
            "score": row["score"],
            "region_points": row["components"]["region"],
            "boundary_points": row["components"]["boundary"],
            "stability_points": row["components"]["stability"],
            "reliability_points": row["components"]["reliability"],
            "p0_robustness_points": row["components"]["p0_robustness"],
            "dice_mean": row["raw"]["dice_mean"],
            "jaccard_mean": row["raw"]["jaccard_mean"],
            "boundary_f1_mean": row["raw"]["boundary_f1_mean"],
            "hd95_normalized_mean": row["raw"]["hd95_normalized_mean"],
            "dice_iqr": row["raw"]["dice_iqr"],
            "technical_failure_rate": row["raw"]["technical_failure_rate"],
            "degenerate_prediction_rate": row["raw"]["degenerate_prediction_rate"],
            "p0_robustness": row["raw"]["p0_robustness"],
            "exclusion_reasons": " | ".join(row["exclusion_reasons"]),
        })
    columns = list(flat[0]) if flat else []
    stream = io.StringIO(newline="")
    if columns:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(flat)
    atomic_write_bytes(path, stream.getvalue().encode("utf-8"))


def _select(args: argparse.Namespace) -> None:
    b1_manifest_path = args.b1_run / "run_manifest.json"
    ablation_manifest_paths = [run / "run_manifest.json" for run in args.ablation_run]
    for path in [b1_manifest_path, *ablation_manifest_paths]:
        if not path.is_file():
            raise SystemExit(f"Missing run manifest: {path}")
        if load_json(path).get("status") != "completed":
            raise SystemExit(f"Run is not completed: {path}")
    b1 = load_json(b1_manifest_path)
    ablations = [load_json(path) for path in ablation_manifest_paths]
    if b1.get("evaluation") != "B1":
        raise SystemExit("--b1-run must be a completed B1 run")
    if any(item.get("evaluation") != "post05_component_ablation" for item in ablations):
        raise SystemExit("Every --ablation-run must be a post-0.5 component ablation run")
    identity = (b1.get("configuration") or {}).get("post05_contract", {}).get("yolo_identity_hash")
    if not identity or any(identity != item.get("yolo_final_identity_hash") for item in ablations):
        raise SystemExit("B1 and all ablation runs must share the same frozen final YOLO identity")
    required_conditions = {"B1_FULL", "B1_NO_HAIR", "B1_NO_YOLO", "B1_NO_FOV", "B1_NO_POST"}
    condition_counts = {condition: sum(condition in item.get("conditions", []) for item in ablations) for condition in required_conditions}
    observed_conditions = {condition for item in ablations for condition in item.get("conditions", [])}
    if observed_conditions != required_conditions or any(count != 1 for count in condition_counts.values()):
        raise SystemExit(
            f"Ablation runs must cover every pre-registered condition exactly once; "
            f"observed={sorted(observed_conditions)} counts={condition_counts}"
        )

    full_rows = load_run_rows(args.b1_run)
    ablation_rows = [row for run in args.ablation_run for row in load_run_rows(run)]
    ranking = choose_top3(
        full_rows,
        ablation_rows,
        max_technical_failure_rate=args.max_technical_failure_rate,
        max_degenerate_rate=args.max_degenerate_rate,
    )
    eligible_rank = 0
    for row in ranking:
        if row["eligible"]:
            eligible_rank += 1
            row["rank"] = eligible_rank
        else:
            row["rank"] = None
    top3 = [row["method_id"] for row in ranking if row["eligible"]][:3]
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_name": "post05_top3_segmenters",
        "score_max": 100,
        "score_formula": {
            "region_35": "17.5*mean(Dice) + 17.5*mean(Jaccard)",
            "boundary_20": "12*mean(BoundaryF1) + 8/(1+mean(HD95_normalized))",
            "stability_15": "15*(1-IQR(Dice))",
            "reliability_15": "15*clip(1-technical_failure_rate-0.75*degenerate_rate,0,1)",
            "p0_robustness_15": "15*clip(mean(Dice across five ablations)-std(Dice across five ablations),0,1)",
        },
        "eligibility_gate": {
            "max_technical_failure_rate": args.max_technical_failure_rate,
            "max_degenerate_prediction_rate": args.max_degenerate_rate,
        },
        "tie_break": "higher total score, then lexicographic method_id; gates applied before rank",
        "b1_run": str(args.b1_run.resolve()),
        "b1_run_manifest_sha256": sha256_file(b1_manifest_path),
        "ablation_runs": [str(run.resolve()) for run in args.ablation_run],
        "ablation_run_manifest_sha256": [sha256_file(path) for path in ablation_manifest_paths],
        "yolo_final_identity_hash": identity,
        "ranking": ranking,
        "top3": top3,
        "top3_resources": {method: (b1.get("resources") or {}).get(method) for method in top3},
        "git": _git_state(),
    }
    payload["identity_hash"] = content_hash(payload)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output / "segmenter_selection.json", payload)
    _write_csv(args.output / "segmenter_selection.csv", ranking)
    lines = [
        "# Selección TOP-3 post Fase 0.5",
        "",
        "El score fue definido antes de MSKCC y suma 100 puntos: región 35, borde 20, estabilidad 15, fiabilidad 15 y robustez P0 15.",
        "",
        "| Rank | Método | Elegible | Score | Región | Borde | Estabilidad | Fiabilidad | Robustez P0 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranking:
        lines.append(
            f"| {row['rank'] or '-'} | {row['method_id']} | {row['eligible']} | {row['score']:.4f} | "
            f"{row['components']['region']:.4f} | {row['components']['boundary']:.4f} | "
            f"{row['components']['stability']:.4f} | {row['components']['reliability']:.4f} | "
            f"{row['components']['p0_robustness']:.4f} |"
        )
    lines.extend(["", "TOP-3 congelable: **" + ", ".join(top3) + "**", ""])
    atomic_write_bytes(args.output / "segmenter_selection.md", ("\n".join(lines)).encode("utf-8"))
    print(json.dumps({"top3": top3, "output": str(args.output), "identity_hash": payload["identity_hash"]}, indent=2))


def _freeze(args: argparse.Namespace) -> None:
    git = _git_state()
    if git["dirty"]:
        raise SystemExit("Scientific freeze refuses a dirty Git tree. Commit the methodology/code first.")
    final_yolo = validate_final_yolo_manifest(args.final_yolo, verify_artifacts=True)
    selection_path = args.selection / "segmenter_selection.json" if args.selection.is_dir() else args.selection
    selection = load_json(selection_path)
    if content_hash({key: value for key, value in selection.items() if key != "identity_hash"}) != selection.get("identity_hash"):
        raise SystemExit("Segmenter selection identity hash is invalid")
    if selection.get("yolo_final_identity_hash") != final_yolo["identity_hash"]:
        raise SystemExit("Selection and final YOLO identities differ")
    top3 = selection.get("top3") or []
    if len(top3) != 3 or len(set(top3)) != 3:
        raise SystemExit("Selection does not contain exactly three unique methods")
    top3_resources = selection.get("top3_resources") or {}
    if set(top3_resources) != set(top3):
        raise SystemExit("Selection does not freeze resources for exactly the TOP-3 methods")
    for method, resource in top3_resources.items():
        if method == "S16":
            continue
        if not isinstance(resource, dict) or not resource.get("available") or not resource.get("sha256"):
            raise SystemExit(f"TOP-3 resource was not available/hashable in B1: {method}")
    b1_manifest = Path(selection["b1_run"]) / "run_manifest.json"
    ablation_manifests = [Path(run) / "run_manifest.json" for run in selection["ablation_runs"]]
    if sha256_file(b1_manifest) != selection["b1_run_manifest_sha256"]:
        raise SystemExit("B1 run manifest changed after TOP-3 selection")
    if [sha256_file(path) for path in ablation_manifests] != selection["ablation_run_manifest_sha256"]:
        raise SystemExit("An ablation run manifest changed after TOP-3 selection")

    payload = {
        "schema_version": 1,
        "status": "frozen_before_mskcc",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": git,
        "final_yolo_manifest": str(args.final_yolo.resolve()),
        "final_yolo_manifest_sha256": sha256_file(args.final_yolo),
        "final_yolo_identity_hash": final_yolo["identity_hash"],
        "selection_manifest": str(selection_path.resolve()),
        "selection_manifest_sha256": sha256_file(selection_path),
        "selection_identity_hash": selection["identity_hash"],
        "top3": top3,
        "top3_resources": top3_resources,
        "ablation_conditions": ["B1_FULL", "B1_NO_HAIR", "B1_NO_YOLO", "B1_NO_FOV", "B1_NO_POST"],
        "mskcc_sampling_contract": {
            "target_n": 100,
            "primary_strata": "MST 1..10, target 10 per tone",
            "fallback_strata": "MST pairs 1-2,3-4,5-6,7-8,9-10, target 20 per pair",
            "one_image_per_lesion": True,
            "max_lesions_per_patient": 2,
            "single_acquisition_mode": True,
            "prefer_complete_variables": ["FST", "MST", "L*", "b*", "ITA"],
        },
        "review_contract": {
            "top3_candidates_blinded": True,
            "main_review_n": 80,
            "independent_reference_n": 20,
            "independent_reference_generator": "SAM candidate from previously investigated family, followed by full human correction",
            "reference_name": "human-corrected reference mask",
        },
        "fairness_variables": {
            "primary": ["MST", "ITA"],
            "secondary": ["FST", "L*"],
            "additional_endpoints": ["YOLO fallback rate", "degenerate prediction rate"],
        },
        "sealed_test_policy": "No downstream MSKCC decision may alter the frozen YOLO, TOP-3, score, or ablation definitions.",
    }
    payload["identity_hash"] = content_hash(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    select = sub.add_parser("select")
    select.add_argument("--b1-run", required=True, type=Path)
    select.add_argument("--ablation-run", required=True, action="append", type=Path)
    select.add_argument("--output", required=True, type=Path)
    select.add_argument("--max-technical-failure-rate", type=float, default=0.01)
    select.add_argument("--max-degenerate-rate", type=float, default=0.10)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--final-yolo", required=True, type=Path)
    freeze.add_argument("--selection", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.action == "select":
        _select(args)
    else:
        _freeze(args)


if __name__ == "__main__":
    main()
