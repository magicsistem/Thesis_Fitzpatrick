#!/usr/bin/env python3
"""Prepare, audit, and one-shot execute the integrity-sealed final test."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import os


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis_fitzpatrick.benchmark import (  # noqa: E402
    atomic_write_json,
    content_hash,
    load_benchmark_methods,
    load_json,
    sha256_file,
    validate_dataset_manifest,
)
from thesis_fitzpatrick.datasets import verify_manifest_files  # noqa: E402


def tracked_tree_identity() -> tuple[dict[str, str], str]:
    """Hash every Git-tracked byte so the seal does not rely on a commit label alone."""
    completed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, check=True, capture_output=True
    )
    paths = sorted(value.decode("utf-8") for value in completed.stdout.split(b"\0") if value)
    identities = {relative: sha256_file(REPO_ROOT / relative) for relative in paths}
    return identities, content_hash(identities)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare = subparsers.add_parser("prepare", help="Audit hashes and write a preparation manifest only")
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--split", required=True, choices=("test",))
    prepare.add_argument("--evaluation", required=True, choices=("B2",))
    prepare.add_argument("--config", required=True, type=Path)
    prepare.add_argument("--data-root", type=Path)
    prepare.add_argument("--manifest", type=Path)
    prepare.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "benchmark_v1" / "sealed")
    prepare.add_argument("--note", required=True, help="Reason for the final evaluation")
    audit = subparsers.add_parser("audit", help="Verify every frozen identity without executing")
    audit.add_argument("--prepared", required=True, type=Path)
    execute = subparsers.add_parser("execute", help="Execute exactly once without intermediate test metrics")
    execute.add_argument("--prepared", required=True, type=Path)
    execute.add_argument("--confirmation", required=True, help="Must equal EJECUTAR TEST SELLADO UNA VEZ")
    return parser.parse_args()


def append_attempt(directory: Path, payload: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "attempts.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), **payload}, ensure_ascii=False, sort_keys=True) + "\n")


def audit_prepared(path: Path) -> dict:
    payload = load_json(path)
    problems = []
    if payload.get("status") not in {"prepared", "executing", "sealed", "failed"}: problems.append("estado inválido")
    current_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    if current_commit != payload.get("git_commit"): problems.append("commit modificado")
    if payload.get("code_files") is not None:
        current_files, current_tree_hash = tracked_tree_identity()
        if current_files != payload["code_files"] or current_tree_hash != payload.get("code_tree_hash"):
            problems.append("código rastreado modificado")
    config_path = Path(payload["configuration_path"])
    if content_hash(load_json(config_path)) != payload.get("configuration_hash"): problems.append("configuración modificada")
    if payload.get("base_config_hash"):
        base_config_path = Path(payload["base_config_path"])
        if not base_config_path.is_file() or sha256_file(base_config_path) != payload["base_config_hash"]:
            problems.append("configuración base modificada")
    manifest_path = Path(payload["dataset_manifest_path"])
    if content_hash(load_json(manifest_path)) != payload.get("dataset_manifest_hash"): problems.append("manifest de datos modificado")
    if payload.get("data_root") and payload.get("dataset_files_verified"):
        try:
            if not verify_manifest_files(load_json(manifest_path), Path(payload["data_root"]))["passed"]:
                problems.append("archivos del dataset modificados")
        except (OSError, ValueError, FileNotFoundError):
            problems.append("archivos del dataset modificados")
    for method_id, identity in payload.get("checkpoints", {}).items():
        for member in identity.get("members", [identity]):
            checkpoint = Path(member["resolved_path"])
            if not checkpoint.is_file() or sha256_file(checkpoint) != member["sha256"]: problems.append(f"checkpoint modificado: {method_id}")
            metadata_path = Path(member["resolved_metadata_path"]) if member.get("resolved_metadata_path") else None
            if metadata_path and (not metadata_path.is_file() or sha256_file(metadata_path) != member["metadata_sha256"]):
                problems.append(f"metadatos de checkpoint modificados: {method_id}")
    if payload.get("status") == "sealed":
        run_root = REPO_ROOT / "results/benchmark_v1/runs" / payload["run_id"]
        for relative, expected in payload.get("result_files", {}).items():
            path = run_root / relative
            if not path.is_file() or sha256_file(path) != expected: problems.append(f"resultado sellado modificado: {relative}")
    return {"passed": not problems, "problems": problems, "status": payload.get("status")}


def main() -> None:
    args = parse_args()
    if args.action in {"audit", "execute"}:
        prepared = load_json(args.prepared)
        directory = args.prepared.parent
        audit = audit_prepared(args.prepared)
        if args.action == "audit":
            append_attempt(directory, {"action": "audit", **audit}); print(json.dumps(audit, indent=2));
            if not audit["passed"]: raise SystemExit(2)
            return
        append_attempt(directory, {"action": "execute_requested", "audit": audit, "status_before": prepared.get("status")})
        if args.confirmation != "EJECUTAR TEST SELLADO UNA VEZ": raise SystemExit("Confirmación exacta incorrecta; intento registrado.")
        if not audit["passed"]: raise SystemExit("Auditoría de integridad fallida; intento registrado.")
        if prepared.get("status") != "prepared" or not prepared.get("execution_locked"): raise SystemExit("La preparación ya fue usada o no está bloqueada; intento registrado.")
        status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout
        if status: raise SystemExit("El árbol dejó de estar limpio; intento registrado.")
        prepared.update({"status": "executing", "execution_locked": False, "execution_started_utc": datetime.now(timezone.utc).isoformat()})
        atomic_write_json(args.prepared, prepared)
        run_id = "sealed-" + prepared["git_commit"][:12] + "-" + prepared["configuration_hash"][:12]
        command = [sys.executable, str(REPO_ROOT / "scripts/benchmark/run_evaluation.py"), "--evaluation", "B2", "--dataset", prepared["dataset_id"], "--split", "test", "--manifest", prepared["dataset_manifest_path"], "--data-root", prepared["data_root"], "--config", prepared["base_config_path"], "--b2-config", prepared["configuration_path"], "--models", ",".join(prepared["methods"]), "--run-id", run_id, "--confirm-run", "--sealed-authority", str(args.prepared)]
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        result_files = {}
        if completed.returncode == 0:
            run_root = REPO_ROOT / "results/benchmark_v1/runs" / run_id
            result_files = {path.relative_to(run_root).as_posix(): sha256_file(path) for path in sorted(run_root.rglob("*")) if path.is_file()}
        prepared.update({"status": "sealed" if completed.returncode == 0 else "failed", "execution_locked": True, "execution_finished_utc": datetime.now(timezone.utc).isoformat(), "run_id": run_id, "returncode": completed.returncode, "result_files": result_files, "result_integrity_hash": content_hash(result_files) if result_files else None, "immutability_note": "Logical one-shot lock plus hashes; local files are not claimed to be physically immutable."})
        atomic_write_json(args.prepared, prepared); append_attempt(directory, {"action": "execution_finished", "status": prepared["status"], "returncode": completed.returncode})
        if completed.returncode: raise SystemExit(completed.returncode)
        print(json.dumps({"status": "sealed", "run_id": run_id}, indent=2)); return
    config = load_json(args.config)
    if config.get("schema_version") != 2 or config.get("evaluation") != "B2":
        raise SystemExit("La configuración sellada debe declarar schema_version=2 y evaluation=B2.")
    if config.get("frozen") is not True:
        raise SystemExit("Configuración no congelada: frozen debe ser true después de completar y validar B2.")
    status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout
    if status:
        raise SystemExit("El test sellado exige un árbol Git limpio; hay cambios locales sin congelar.")
    methods = load_benchmark_methods(REPO_ROOT / "configs" / "segmentation_models.json")
    expected = {method["method_id"] for method in methods if method["kind"] == "neural"}
    checkpoints = config.get("b2_checkpoints", {})
    if set(checkpoints) != expected:
        missing = sorted(expected - set(checkpoints))
        extra = sorted(set(checkpoints) - expected)
        raise SystemExit(f"Checkpoints B2 incompletos. Faltan: {missing}; sobran: {extra}")
    checkpoint_audit = {}
    for method_id, identity in checkpoints.items():
        members = identity.get("members") or [identity]
        if len(members) not in {1, 5}: raise SystemExit(f"B2 {method_id} no contiene checkpoint final ni cinco folds")
        audited_members = []
        for member in members:
            path = Path(member.get("path", ""))
            if not path.is_absolute(): path = REPO_ROOT / path
            if not path.is_file(): raise SystemExit(f"Falta checkpoint B2 {method_id}: {path}")
            actual = sha256_file(path)
            if actual != member.get("sha256"): raise SystemExit(f"Hash de checkpoint modificado para {method_id}")
            metadata_path = Path(member.get("metadata_path", ""))
            if not metadata_path.is_absolute(): metadata_path = REPO_ROOT / metadata_path
            if not metadata_path.is_file() or sha256_file(metadata_path) != member.get("metadata_sha256"):
                raise SystemExit(f"Metadatos B2 ausentes o modificados para {method_id}: {metadata_path}")
            metadata = load_json(metadata_path)
            contract = metadata.get("training_contract", {})
            if metadata.get("schema_version") != 2 or metadata.get("status") != "completed" or metadata.get("protocol") != "B2" or metadata.get("method_id") != method_id or metadata.get("checkpoint_sha256") != actual or contract.get("method_id") != method_id or contract.get("fold") != member.get("fold"):
                raise SystemExit(f"Metadatos B2 incompatibles para {method_id}: {metadata_path}")
            audited_members.append({"path": member["path"], "resolved_path": str(path.resolve()), "sha256": actual, "metadata_path": member["metadata_path"], "resolved_metadata_path": str(metadata_path.resolve()), "metadata_sha256": member["metadata_sha256"]})
        checkpoint_audit[method_id] = {"aggregation": identity.get("aggregation", "single"), "members": audited_members}
    data_root = args.data_root or REPO_ROOT / "data" / "raw" / args.dataset
    manifest_path = args.manifest or data_root / "manifests" / f"{args.dataset}_test.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Falta el manifest sellado: {manifest_path}")
    dataset_manifest = load_json(manifest_path)
    audit = validate_dataset_manifest(dataset_manifest, data_root=data_root, require_files=True)
    integrity = verify_manifest_files(dataset_manifest, data_root)
    if audit["items"] != 1000:
        raise SystemExit(f"ISIC 2018 test debe contener 1000 imágenes; manifest actual: {audit['items']}")
    if dataset_manifest.get("dataset_id") != args.dataset or dataset_manifest.get("split") != "test" or dataset_manifest.get("complete") is not True or dataset_manifest.get("integrity_errors"):
        raise SystemExit("El manifest ISIC 2018 test no está completo, no corresponde al test o contiene errores")
    if not integrity["passed"]:
        raise SystemExit("Falló la verificación de bytes/checksums del test sellado")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    code_files, code_tree_hash = tracked_tree_identity()
    base_config_path = (REPO_ROOT / config.get("base_config", "configs/benchmark/default.json")).resolve()
    if not base_config_path.is_file() or sha256_file(base_config_path) != config.get("base_config_sha256"):
        raise SystemExit("La configuración base no coincide con el hash congelado B2")
    prepared = {
        "schema_version": 1,
        "status": "prepared",
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": args.dataset,
        "split": args.split,
        "evaluation": args.evaluation,
        "note": args.note,
        "git_commit": commit,
        "code_files": code_files,
        "code_tree_hash": code_tree_hash,
        "configuration_path": str(args.config),
        "configuration_hash": content_hash(config),
        "dataset_manifest_path": str(manifest_path),
        "data_root": str(data_root.resolve()),
        "dataset_files_verified": True,
        "base_config_path": str(base_config_path),
        "base_config_hash": sha256_file(base_config_path),
        "dataset_manifest_hash": audit["manifest_hash"],
        "image_count": audit["items"],
        "checkpoints": checkpoint_audit,
        "methods": sorted(expected) + ["S16"],
        "execution_locked": True,
        "unlock_requirements": "A separate execute action requires a clean tree, unchanged hashes, and the exact confirmation phrase.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / f"prepared-{commit[:12]}-{prepared['configuration_hash'][:12]}.json"
    if output.exists():
        raise SystemExit(f"No se sobrescribe una preparación existente: {output}")
    atomic_write_json(output, prepared)
    print(json.dumps({"status": "prepared", "manifest": str(output), "execution_locked": True}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
