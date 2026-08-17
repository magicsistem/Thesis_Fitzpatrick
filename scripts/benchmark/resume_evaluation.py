#!/usr/bin/env python3
"""Safely repair/finalize an existing A/B1 evaluation without repeating valid predictions."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "benchmark"))

from thesis_fitzpatrick.benchmark import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_json,
    content_hash,
    is_fatal_backend_failure,
    load_json,
)
from thesis_fitzpatrick.reporting import write_report  # noqa: E402
import run_evaluation as evaluation_cli  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--reuse-p0-cache", action="store_true")
    parser.add_argument("--confirm-repair", action="store_true")
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        atomic_write_bytes(path, b"")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_bytes(path, stream.getvalue().encode("utf-8"))


def _failure_code(payload: dict[str, Any]) -> str | None:
    return payload.get("failure_code") or (payload.get("backend") or {}).get("failure_code")


def _git_is_ancestor(ancestor: str, descendant: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def validate_contract(
    run_manifest: dict[str, Any],
    *,
    dataset_manifest: dict[str, Any],
    config: dict[str, Any],
    current_git: dict[str, Any],
    current_resources: dict[str, Any],
) -> None:
    if run_manifest.get("evaluation") not in {"A", "B1"}:
        raise ValueError("Only A/B1 evaluation runs can be repaired by this command")
    if run_manifest.get("dataset") != dataset_manifest.get("dataset_id"):
        raise ValueError("Dataset id changed")
    if run_manifest.get("split") != dataset_manifest.get("split"):
        raise ValueError("Dataset split changed")
    image_ids = [item.get("image_id") for item in dataset_manifest.get("items", [])]
    if run_manifest.get("images") != image_ids:
        raise ValueError("Image list/order changed")
    if run_manifest.get("dataset_manifest_hash") != content_hash(dataset_manifest):
        raise ValueError("Dataset manifest hash changed")
    if run_manifest.get("configuration_hash") != content_hash(config):
        raise ValueError("Benchmark configuration changed")
    if content_hash(run_manifest.get("configuration", {})) != content_hash(config):
        raise ValueError("Stored configuration payload differs from current config")
    if run_manifest.get("resources") != current_resources:
        raise ValueError("Checkpoint/resource identities changed")
    original_git = run_manifest.get("git") or {}
    original_commit = original_git.get("commit")
    current_commit = current_git.get("commit")
    if original_git.get("dirty"):
        raise ValueError("Original run had a dirty working tree")
    if current_git.get("dirty"):
        raise ValueError("Current working tree is dirty; commit the repair before running it")
    if not original_commit or not current_commit:
        raise ValueError("Missing Git commit identity")
    if original_commit != current_commit and not _git_is_ancestor(original_commit, current_commit):
        raise ValueError("Original run commit is not an ancestor of the current repair commit")


def scan_run(
    run_directory: Path,
    run_manifest: dict[str, Any],
    *,
    store_web_previews: bool,
) -> dict[str, Any]:
    metrics_rows: list[dict[str, Any]] = []
    technical_failures: list[dict[str, Any]] = []
    quality_flags: list[dict[str, Any]] = []
    fallbacks: list[dict[str, Any]] = []
    results_seen = 0
    for image_id in run_manifest["images"]:
        for method_id in run_manifest["methods"]:
            directory = run_directory / "predictions" / method_id / image_id
            result_path = directory / "result.json"
            if not result_path.is_file():
                technical_failures.append({"image_id": image_id, "method_id": method_id, "failure_code": "missing_result", "warnings": "result.json is missing"})
                continue
            try:
                payload = load_json(result_path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                technical_failures.append({"image_id": image_id, "method_id": method_id, "failure_code": "invalid_result_json", "warnings": str(exc)})
                continue
            results_seen += 1
            expected = {
                "evaluation": run_manifest["evaluation"],
                "dataset_id": run_manifest["dataset"],
                "split": run_manifest["split"],
                "method_id": method_id,
                "image_id": image_id,
            }
            if any(payload.get(key) != value for key, value in expected.items()):
                technical_failures.append({"image_id": image_id, "method_id": method_id, "failure_code": "result_contract_mismatch", "warnings": "result identity differs from run contract"})
                continue
            code = _failure_code(payload)
            required = ["native_mask.png", "final_mask.png", "clean_skin_mask.png"]
            if store_web_previews:
                required.append("overlay.jpg")
            missing = [name for name in required if not (directory / name).is_file()]
            if missing:
                technical_failures.append({
                    "image_id": image_id,
                    "method_id": method_id,
                    "failure_code": code if is_fatal_backend_failure(code) else "missing_artifact",
                    "warnings": "missing: " + ", ".join(missing),
                })
                continue
            if is_fatal_backend_failure(code):
                technical_failures.append({
                    "image_id": image_id,
                    "method_id": method_id,
                    "failure_code": code,
                    "warnings": " | ".join((payload.get("backend") or {}).get("warnings") or []),
                })
                continue
            metrics = payload.get("metrics")
            if not isinstance(metrics, dict):
                technical_failures.append({"image_id": image_id, "method_id": method_id, "failure_code": "missing_metrics", "warnings": "validation result has no metrics"})
                continue
            metrics_rows.append({
                "image_id": image_id,
                "method_id": method_id,
                **{key: value for key, value in metrics.items() if isinstance(value, (int, float)) or value is None},
            })
            flags = metrics.get("flags") or {}
            reasons = []
            if code:
                reasons.append(code)
            if flags.get("prediction_empty"):
                reasons.append("prediction_empty")
            if flags.get("prediction_nearly_complete"):
                reasons.append("prediction_nearly_complete")
            if reasons:
                quality_flags.append({
                    "image_id": image_id,
                    "method_id": method_id,
                    "quality_flags": " | ".join(dict.fromkeys(reasons)),
                    "warnings": " | ".join((payload.get("backend") or {}).get("warnings") or []),
                })
            if payload.get("p0_fallback_used"):
                fallbacks.append({"image_id": image_id, "method_id": method_id, "type": "yolo_full_fov"})
    return {
        "metrics_rows": metrics_rows,
        "technical_failures": technical_failures,
        "quality_flags": quality_flags,
        "fallbacks": fallbacks,
        "counts": {
            "expected": len(run_manifest["images"]) * len(run_manifest["methods"]),
            "results": results_seen,
            "scientific_metrics": len(metrics_rows),
            "technical_failures": len(technical_failures),
            "quality_flagged_predictions": len(quality_flags),
        },
    }


def _archive_run_level_state(run_directory: Path, attempt_root: Path) -> None:
    attempt_root.mkdir(parents=True, exist_ok=False)
    for name in (
        "run_manifest.json",
        "metrics_per_image.csv",
        "failures.csv",
        "quality_flags.csv",
        "metrics_summary.csv",
        "metrics_summary.md",
        "statistical_comparisons.csv",
        "report.json",
    ):
        source = run_directory / name
        if source.is_file():
            shutil.copy2(source, attempt_root / name)


def _repair_one(
    *,
    failure: dict[str, Any],
    args: argparse.Namespace,
    run_directory: Path,
    run_manifest: dict[str, Any],
    artifact_root: Path,
    attempt_root: Path,
) -> dict[str, Any]:
    method_id, image_id = failure["method_id"], failure["image_id"]
    repair_id = f"repair-{run_directory.name}-{method_id}-{image_id}-{attempt_root.name}"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "benchmark" / "run_evaluation.py"),
        "--evaluation", run_manifest["evaluation"],
        "--dataset", run_manifest["dataset"],
        "--split", run_manifest["split"],
        "--models", method_id,
        "--manifest", str(args.manifest),
        "--data-root", str(args.data_root),
        "--config", str(args.config),
        "--artifact-root", str(artifact_root),
        "--run-id", repair_id,
        "--image-ids", image_id,
        "--warmup", str(args.warmup),
        "--repetitions", str(args.repetitions),
        "--confirm-run",
    ]
    if args.reuse_p0_cache:
        command.append("--reuse-p0-cache")
    process = subprocess.run(command, cwd=REPO_ROOT)
    repair_run = artifact_root / "runs" / repair_id
    if process.returncode != 0:
        raise RuntimeError(f"Repair execution failed for {method_id}/{image_id}; preserved at {repair_run}")
    source = repair_run / "predictions" / method_id / image_id
    payload = load_json(source / "result.json")
    code = _failure_code(payload)
    if is_fatal_backend_failure(code):
        raise RuntimeError(f"Repair remained a technical failure for {method_id}/{image_id}: {code}")
    for name in ("native_mask.png", "final_mask.png", "clean_skin_mask.png", "result.json"):
        if not (source / name).is_file():
            raise RuntimeError(f"Repair output missing {name} for {method_id}/{image_id}")
    target = run_directory / "predictions" / method_id / image_id
    old_archive = attempt_root / "predictions_before" / method_id / image_id
    if target.exists():
        old_archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(target, old_archive)
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    repair_archive = attempt_root / "repair_runs" / repair_id
    repair_archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(repair_run), str(repair_archive))
    return {
        "image_id": image_id,
        "method_id": method_id,
        "previous_failure_code": failure.get("failure_code"),
        "repair_archive": str(repair_archive.relative_to(run_directory)),
        "result_failure_code": code,
    }


def main() -> None:
    args = parse_args()
    run_directory = args.run_directory.resolve()
    if run_directory.parent.name != "runs" or not (run_directory / "run_manifest.json").is_file():
        raise SystemExit("--run-directory must be an existing benchmark runs/<run-id> directory")
    run_manifest = load_json(run_directory / "run_manifest.json")
    if run_manifest.get("run_id") != run_directory.name:
        raise SystemExit("Run directory name does not match run_manifest.run_id")
    if run_manifest.get("status") == "completed":
        print(json.dumps({"run_id": run_directory.name, "status": "completed", "message": "nothing to repair"}, indent=2))
        return
    dataset_manifest = load_json(args.manifest)
    config = load_json(args.config)
    methods = evaluation_cli._select_methods(",".join(run_manifest.get("methods") or []))
    resources = {method["method_id"]: evaluation_cli._checkpoint(method) for method in methods}
    current_git = evaluation_cli._git_state()
    try:
        validate_contract(
            run_manifest,
            dataset_manifest=dataset_manifest,
            config=config,
            current_git=current_git,
            current_resources=resources,
        )
    except ValueError as exc:
        raise SystemExit(f"Unsafe repair refused: {exc}") from exc

    artifact_root = run_directory.parent.parent
    initial = scan_run(run_directory, run_manifest, store_web_previews=config.get("store_web_previews", True))
    failures = initial["technical_failures"]
    if failures and not args.confirm_repair:
        raise SystemExit(f"{len(failures)} technical failures require --confirm-repair before rerun")

    attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    attempt_root = run_directory / "resume_attempts" / attempt_id
    _archive_run_level_state(run_directory, attempt_root)
    history = {
        "attempt_id": attempt_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "from_status": run_manifest.get("status"),
        "from_fatal_error": run_manifest.get("fatal_error"),
        "from_git": run_manifest.get("git"),
        "to_git": current_git,
        "initial_counts": initial["counts"],
        "repairs": [],
    }
    run_manifest.setdefault("resume_history", []).append(history)
    run_manifest["status"] = "repairing"
    run_manifest["last_execution_git"] = current_git
    atomic_write_json(run_directory / "run_manifest.json", run_manifest)

    try:
        for failure in failures:
            history["repairs"].append(_repair_one(
                failure=failure,
                args=args,
                run_directory=run_directory,
                run_manifest=run_manifest,
                artifact_root=artifact_root,
                attempt_root=attempt_root,
            ))
            atomic_write_json(run_directory / "run_manifest.json", run_manifest)

        final = scan_run(run_directory, run_manifest, store_web_previews=config.get("store_web_previews", True))
        _write_csv(run_directory / "metrics_per_image.csv", final["metrics_rows"])
        _write_csv(run_directory / "failures.csv", final["technical_failures"])
        _write_csv(run_directory / "quality_flags.csv", final["quality_flags"])
        run_manifest["failures"] = final["technical_failures"]
        run_manifest["quality_flags"] = final["quality_flags"]
        run_manifest["fallbacks"] = final["fallbacks"]
        run_manifest["result_counts"] = final["counts"]
        history["final_counts"] = final["counts"]
        if final["technical_failures"]:
            raise RuntimeError(f"{len(final['technical_failures'])} technical failures remain after repair")
        write_report(
            run_directory,
            repetitions=config["metrics"]["bootstrap_repetitions"],
            seed=config["seed"],
        )
        run_manifest["status"] = "completed"
        run_manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        run_manifest.pop("fatal_error", None)
        history["status"] = "completed"
        history["finished_utc"] = datetime.now(timezone.utc).isoformat()
    except BaseException as exc:
        run_manifest["status"] = "failed"
        run_manifest["fatal_error"] = f"{type(exc).__name__}: {exc}"
        history["status"] = "failed"
        history["finished_utc"] = datetime.now(timezone.utc).isoformat()
        raise
    finally:
        atomic_write_json(run_directory / "run_manifest.json", run_manifest)

    print(json.dumps({
        "run_id": run_directory.name,
        "status": run_manifest["status"],
        "initial_technical_failures": len(failures),
        "repairs": len(history["repairs"]),
        "quality_flagged_predictions": final["counts"]["quality_flagged_predictions"],
        "scientific_metrics": final["counts"]["scientific_metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()
