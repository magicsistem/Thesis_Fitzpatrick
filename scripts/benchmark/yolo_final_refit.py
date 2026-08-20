#!/usr/bin/env python3
"""Prepare, train, freeze, and configure the single post-CV YOLO final refit.

The five historical YOLO folds stay untouched and are used only for out-of-fold
estimation and hyperparameter selection.  This script trains one new detector on
all authorized ``train_disjoint`` images, using only CV-frozen hyperparameters.
No held-out development image and no sealed-test image is added to the refit.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis_fitzpatrick.benchmark import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_json,
    content_hash,
    load_json,
    sha256_file,
)
from thesis_fitzpatrick.post05 import (  # noqa: E402
    FINAL_YOLO_ROLE,
    FINAL_YOLO_SCHEMA_VERSION,
    build_final_b1_config,
    scaled_final_batches,
    validate_final_yolo_manifest,
)
from thesis_fitzpatrick.yolo import (  # noqa: E402
    bbox_from_mask,
    bbox_to_darknet,
    darknet_weights_iteration,
    patch_yolov3_cfg,
    perturb_bbox,
    validate_frozen_yolo_set,
)


def _parse_data(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "=" in raw and not raw.lstrip().startswith("#"):
            key, value = raw.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _cfg_int(path: Path, name: str) -> int:
    match = re.search(rf"(?m)^{re.escape(name)}\s*=\s*(\d+)\s*$", path.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"Missing {name} in Darknet cfg: {path}")
    return int(match.group(1))


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=False, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, check=False, capture_output=True, text=True).stdout.strip())
    return {"commit": commit or None, "dirty": dirty}


def _prepare(args: argparse.Namespace) -> None:
    manifest = load_json(args.manifest)
    folds = load_json(args.folds)
    if manifest.get("dataset_id") != "isic2018_task1" or manifest.get("split") != "train":
        raise SystemExit("Final YOLO refit accepts only the disjoint ISIC 2018 training manifest")
    items = manifest.get("items") or []
    if not items:
        raise SystemExit("Training manifest is empty")
    ids = [str(item["image_id"]) for item in items]
    if len(ids) != len(set(ids)):
        raise SystemExit("Training manifest contains duplicate image_id values")
    fold_union = set().union(*(set(entry.get("train_ids", [])) | set(entry.get("validation_ids", [])) for entry in folds.get("folds", [])))
    if set(ids) != fold_union:
        raise SystemExit("Final refit manifest does not match the exact image universe used by the five CV folds")

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Final YOLO directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir()
    (output / "labels").mkdir()
    (output / "backup").mkdir()
    (output / "training").mkdir()

    max_batches = scaled_final_batches(len(items), folds, args.cv_max_batches)
    cfg = output / "lesion-yolov3-final.cfg"
    patch_yolov3_cfg(
        args.source_cfg,
        cfg,
        batch=args.batch,
        subdivisions=args.subdivisions,
        width=args.input_size,
        height=args.input_size,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        max_batches=max_batches,
    )

    import random

    rng = random.Random(args.seed)
    train_paths: list[str] = []
    failures: list[dict[str, str]] = []
    for item in sorted(items, key=lambda row: str(row["image_id"])):
        image_id = str(item["image_id"])
        masks = item.get("mask_paths") or []
        if not masks:
            failures.append({"image_id": image_id, "reason": "training mask missing"})
            continue
        image_path = args.data_root / item["image_path"]
        mask_path = args.data_root / masks[0]
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or image.shape[:2] != mask.shape:
            failures.append({"image_id": image_id, "reason": "invalid or dimension-mismatched image/mask"})
            continue
        try:
            box = bbox_from_mask(mask)
        except ValueError as exc:
            failures.append({"image_id": image_id, "reason": str(exc)})
            continue
        if args.box_perturbation:
            box = perturb_bbox(box, image.shape[1], image.shape[0], args.box_perturbation, rng)
        values = bbox_to_darknet(box, image.shape[1], image.shape[0])
        label = output / "labels" / f"{image_id}.txt"
        atomic_write_bytes(label, ("0 " + " ".join(f"{value:.10f}" for value in values) + "\n").encode())
        linked = output / "images" / f"{image_id}{image_path.suffix.lower()}"
        linked.symlink_to(image_path.resolve())
        train_paths.append(str(linked.absolute()))

    if failures:
        atomic_write_json(output / "prepare_failures.json", failures)
        raise SystemExit(f"Final YOLO preparation rejected {len(failures)} images; inspect prepare_failures.json")
    if len(train_paths) != len(items):
        raise SystemExit("Prepared training count does not match manifest count")

    train_txt = output / "train.txt"
    atomic_write_bytes(train_txt, ("\n".join(train_paths) + "\n").encode())
    # Darknet parses a valid path from .data even when -map is absent.  Point it
    # at train.txt solely as a parser-safe placeholder and explicitly forbid
    # interpreting it as validation.  The final-refit training command never
    # requests -map and no parameter is selected from these data.
    names = output / "lesion.names"
    atomic_write_bytes(names, b"lesion\n")
    data_file = output / "lesion.data"
    atomic_write_bytes(
        data_file,
        (
            f"classes = 1\ntrain = {train_txt.resolve()}\nvalid = {train_txt.resolve()}\n"
            f"names = {names.resolve()}\nbackup = {(output / 'backup').resolve()}\n"
        ).encode(),
    )
    prepare = {
        "schema_version": 1,
        "model_role": FINAL_YOLO_ROLE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": manifest["dataset_id"],
        "source_split": manifest["split"],
        "training_count": len(train_paths),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "folds_path": str(args.folds.resolve()),
        "folds_sha256": sha256_file(args.folds),
        "cv_max_batches": args.cv_max_batches,
        "final_max_batches": max_batches,
        "exposure_rule": "round(cv_max_batches * N_final / mean(N_fold_train))",
        "cv_train_sizes": [len(entry["train_ids"]) for entry in folds["folds"]],
        "source_cfg_path": str(args.source_cfg.resolve()),
        "source_cfg_sha256": sha256_file(args.source_cfg),
        "cfg_path": str(cfg.resolve()),
        "cfg_sha256": sha256_file(cfg),
        "data_path": str(data_file.resolve()),
        "data_sha256": sha256_file(data_file),
        "train_list_sha256": sha256_file(train_txt),
        "label_policy": "bbox from authorized lesion mask; fixed-seed perturbation applied only to training labels",
        "box_perturbation_fraction": args.box_perturbation,
        "seed": args.seed,
        "validation_policy": "no held-out data used; Darknet -map disabled; valid=train is parser-only",
        "sealed_test_used": False,
        "git": _git_state(),
    }
    atomic_write_json(output / "prepare_manifest.json", prepare)
    print(json.dumps(prepare, indent=2))


def _training_contract(root: Path, darknet: Path, initial_weights: Path) -> dict[str, Any]:
    prepare = load_json(root / "prepare_manifest.json")
    cfg = Path(prepare["cfg_path"])
    data_file = Path(prepare["data_path"])
    data = _parse_data(data_file)
    train = Path(data["train"])
    names = Path(data["names"])
    return {
        "model_role": FINAL_YOLO_ROLE,
        "darknet_path": str(darknet.resolve()),
        "darknet_sha256": sha256_file(darknet),
        "initial_weights_path": str(initial_weights.resolve()),
        "initial_weights_sha256": sha256_file(initial_weights),
        "cfg_path": str(cfg.resolve()),
        "cfg_sha256": sha256_file(cfg),
        "data_path": str(data_file.resolve()),
        "data_sha256": sha256_file(data_file),
        "train_path": str(train.resolve()),
        "train_sha256": sha256_file(train),
        "names_path": str(names.resolve()),
        "names_sha256": sha256_file(names),
        "batch": _cfg_int(cfg, "batch"),
        "max_batches": _cfg_int(cfg, "max_batches"),
        "training_count": int(prepare["training_count"]),
        "prepare_manifest_sha256": sha256_file(root / "prepare_manifest.json"),
        "git_commit": _git_state()["commit"],
    }


def _checkpoint_candidates(backup: Path, cfg: Path, batch: int, max_batches: int) -> list[tuple[int, Path]]:
    candidates: list[tuple[int, Path]] = []
    pattern = re.compile(rf"{re.escape(cfg.stem)}_(\d+)\.weights$")
    for path in backup.iterdir():
        if not path.is_file() or not (path.name == f"{cfg.stem}.backup" or pattern.fullmatch(path.name)):
            continue
        try:
            iteration = darknet_weights_iteration(path, batch)
        except (OSError, ValueError):
            continue
        if 0 < iteration < max_batches:
            candidates.append((iteration, path))
    return sorted(candidates, key=lambda pair: (pair[0], pair[1].name == f"{cfg.stem}.backup"))


def _validate_training_state(root: Path) -> dict[str, Any]:
    state_path = root / "training" / "training_state.json"
    state = load_json(state_path)
    if state.get("model_role") != FINAL_YOLO_ROLE or state.get("status") != "completed":
        raise ValueError("Final YOLO training is not completed")
    contract = state.get("contract") or {}
    for name in ("darknet", "initial_weights", "cfg", "data", "train", "names"):
        path = Path(contract[f"{name}_path"])
        if not path.is_file() or sha256_file(path) != contract[f"{name}_sha256"]:
            raise ValueError(f"Final YOLO training artifact changed: {path}")
    final = Path(state["final_weights_path"])
    if not final.is_file() or sha256_file(final) != state.get("final_weights_sha256"):
        raise ValueError("Final YOLO weights are missing or changed")
    if darknet_weights_iteration(final, int(contract["batch"])) < int(contract["max_batches"]):
        raise ValueError("Final YOLO weights do not reach max_batches")
    return state


def _train(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    for path in (root / "prepare_manifest.json", args.darknet, args.initial_weights):
        if not path.is_file():
            raise SystemExit(f"Missing prerequisite: {path}")
    contract = _training_contract(root, args.darknet, args.initial_weights)
    state_path = root / "training" / "training_state.json"
    log = root / "training" / "darknet.log"
    cfg = Path(contract["cfg_path"])
    data = Path(contract["data_path"])
    backup = Path(_parse_data(data)["backup"])
    backup.mkdir(parents=True, exist_ok=True)
    final_weights = backup / f"{cfg.stem}_final.weights"

    old_state = load_json(state_path) if state_path.is_file() else None
    if old_state and old_state.get("status") == "completed":
        validated = _validate_training_state(root)
        print(json.dumps({**validated, "reused": True}, indent=2))
        return
    if final_weights.exists():
        raise SystemExit("Final weights exist without a valid completed state; inspect before continuing")

    resume: Path | None = args.resume
    if resume is None and args.auto_resume:
        if old_state:
            old_contract = dict(old_state.get("contract") or {})
            if old_contract != contract:
                raise SystemExit(
                    "Existing final-refit checkpoints belong to a different training contract/Git commit; "
                    "do not resume across code changes without an explicit audit"
                )
        candidates = _checkpoint_candidates(backup, cfg, int(contract["batch"]), int(contract["max_batches"]))
        if candidates:
            resume = candidates[-1][1]
        elif old_state and old_state.get("status") in {"running", "interrupted", "failed"}:
            print("No valid resumable checkpoint found; restarting from bootstrap weights", file=sys.stderr)
    if resume is not None:
        resume = resume.resolve()
        if resume.parent != backup.resolve():
            raise SystemExit("Resume weights must live inside the final-refit backup directory")
        iteration = darknet_weights_iteration(resume, int(contract["batch"]))
        if not 0 < iteration < int(contract["max_batches"]):
            raise SystemExit(f"Resume checkpoint iteration is not valid: {iteration}")
    else:
        iteration = 0

    command = [
        str(args.darknet.resolve()),
        "detector",
        "train",
        str(data),
        str(cfg),
        str((resume or args.initial_weights).resolve()),
        "-dont_show",
    ]
    attempt = int((old_state or {}).get("attempt", 0)) + 1
    attempts = root / "training" / "attempts"
    attempts.mkdir(exist_ok=True)
    if log.exists() and log.stat().st_size:
        archived = attempts / f"darknet-attempt-{attempt - 1}.log"
        if not archived.exists():
            os.replace(log, archived)
    state: dict[str, Any] = {
        "schema_version": 1,
        "model_role": FINAL_YOLO_ROLE,
        "status": "running",
        "attempt": attempt,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "contract_hash": content_hash(contract),
        "command": command,
        "resume_weights_path": str(resume) if resume else None,
        "resume_iteration": iteration,
        "validation_policy": "Darknet -map disabled; no validation/test metrics used during final refit",
    }
    atomic_write_json(state_path, state)

    process: subprocess.Popen[bytes] | None = None
    received_signal: list[int] = []

    def on_signal(signum: int, _frame: Any) -> None:
        received_signal.append(signum)
        state.update({"status": "interrupted", "signal": signum, "signal_utc": datetime.now(timezone.utc).isoformat()})
        atomic_write_json(state_path, state)
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)}
    started = time.monotonic()
    try:
        for sig in handlers:
            signal.signal(sig, on_signal)
        with log.open("wb") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            state["darknet_pid"] = process.pid
            atomic_write_json(state_path, state)
            returncode = process.wait()
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)

    state.update(
        {
            "returncode": returncode,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "training_seconds": time.monotonic() - started,
        }
    )
    if received_signal:
        state.update({"status": "interrupted", "termination_cause": f"signal-{received_signal[-1]}"})
        atomic_write_json(state_path, state)
        raise SystemExit(75)
    if returncode != 0:
        state.update({"status": "failed", "termination_cause": f"darknet-returncode-{returncode}"})
        atomic_write_json(state_path, state)
        raise SystemExit(returncode or 1)
    if not final_weights.is_file():
        state.update({"status": "failed", "termination_cause": "missing-final-weights"})
        atomic_write_json(state_path, state)
        raise SystemExit("Darknet returned success but final weights are missing")
    final_iteration = darknet_weights_iteration(final_weights, int(contract["batch"]))
    if final_iteration < int(contract["max_batches"]):
        state.update({"status": "failed", "termination_cause": f"final-iteration-{final_iteration}"})
        atomic_write_json(state_path, state)
        raise SystemExit(f"Final weights reached {final_iteration}, expected {contract['max_batches']}")
    state.update(
        {
            "status": "completed",
            "termination_cause": "max_batches_reached",
            "final_weights_path": str(final_weights.resolve()),
            "final_weights_bytes": final_weights.stat().st_size,
            "final_weights_sha256": sha256_file(final_weights),
            "observed_iteration": final_iteration,
        }
    )
    atomic_write_json(state_path, state)
    _validate_training_state(root)
    print(json.dumps(state, indent=2))


def _freeze(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    cv = validate_frozen_yolo_set(args.cv_frozen_root.resolve())
    threshold_sets = {
        (
            float(item["thresholds"]["confidence_threshold"]),
            float(item["thresholds"]["nms_threshold"]),
            float(item["thresholds"]["margin_fraction"]),
        )
        for item in cv
    }
    if len(threshold_sets) != 1:
        raise SystemExit(
            "The five CV folds do not have unanimous confidence/NMS/margin thresholds. "
            "Freeze a pre-registered aggregation rule before continuing; do not choose after seeing final-refit results."
        )
    confidence, nms, margin = next(iter(threshold_sets))
    state = _validate_training_state(root)
    prepare = load_json(root / "prepare_manifest.json")
    cfg = Path(state["contract"]["cfg_path"])
    weights = Path(state["final_weights_path"])
    payload = {
        "schema_version": FINAL_YOLO_SCHEMA_VERSION,
        "status": "frozen",
        "architecture": "YOLOv3-Darknet53",
        "model_role": FINAL_YOLO_ROLE,
        "fold": None,
        "cfg_path": str(cfg.resolve()),
        "cfg_sha256": sha256_file(cfg),
        "weights_path": str(weights.resolve()),
        "weights_sha256": sha256_file(weights),
        "training_state_path": str((root / "training" / "training_state.json").resolve()),
        "training_state_sha256": sha256_file(root / "training" / "training_state.json"),
        "training_count": int(prepare["training_count"]),
        "max_batches": int(state["contract"]["max_batches"]),
        "observed_iteration": int(state["observed_iteration"]),
        "thresholds": {
            "confidence_threshold": confidence,
            "nms_threshold": nms,
            "margin_fraction": margin,
        },
        "threshold_selection_policy": "unanimous five-fold CV consensus; final refit did not re-tune thresholds",
        "cv_frozen_root": str(args.cv_frozen_root.resolve()),
        "cv_fold_identity_hashes": [item["identity_hash"] for item in cv],
        "training_manifest_path": prepare["manifest_path"],
        "training_manifest_sha256": prepare["manifest_sha256"],
        "sealed_test_used": False,
        "heldout_development_used_for_training": False,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_state(),
    }
    payload["identity_hash"] = content_hash(payload)
    output = args.output or (root / "frozen.json")
    payload["manifest_path"] = str(output.resolve())
    # manifest_path is descriptive provenance and therefore part of the identity.
    payload.pop("identity_hash")
    payload["identity_hash"] = content_hash(payload)
    atomic_write_json(output, payload)
    validate_final_yolo_manifest(output, verify_artifacts=True)
    print(json.dumps(payload, indent=2))


def _configure(args: argparse.Namespace) -> None:
    frozen = validate_final_yolo_manifest(args.frozen, verify_artifacts=True)
    base = load_json(args.base_config)
    config = build_final_b1_config(base, frozen, artifact_root=args.artifact_root)
    config["p0"]["yolo"]["final_frozen_manifest_path"] = str(args.frozen.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, config)
    print(json.dumps({
        "output": str(args.output),
        "final_yolo_identity_hash": frozen["identity_hash"],
        "weights_sha256": frozen["weights_sha256"],
        "thresholds": frozen["thresholds"],
    }, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--manifest", required=True, type=Path)
    prepare.add_argument("--data-root", required=True, type=Path)
    prepare.add_argument("--folds", required=True, type=Path)
    prepare.add_argument("--source-cfg", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--seed", type=int, default=20260806)
    prepare.add_argument("--cv-max-batches", type=int, default=6000)
    prepare.add_argument("--batch", type=int, default=64)
    prepare.add_argument("--subdivisions", type=int, default=16)
    prepare.add_argument("--input-size", type=int, default=512)
    prepare.add_argument("--learning-rate", type=float, default=0.001)
    prepare.add_argument("--momentum", type=float, default=0.9)
    prepare.add_argument("--weight-decay", type=float, default=0.0005)
    prepare.add_argument("--box-perturbation", type=float, default=0.05)

    train = sub.add_parser("train")
    train.add_argument("--root", required=True, type=Path)
    train.add_argument("--darknet", required=True, type=Path)
    train.add_argument("--initial-weights", required=True, type=Path)
    train.add_argument("--resume", type=Path)
    train.add_argument("--auto-resume", action="store_true")
    train.add_argument("--confirm-training", action="store_true")

    freeze = sub.add_parser("freeze")
    freeze.add_argument("--root", required=True, type=Path)
    freeze.add_argument("--cv-frozen-root", required=True, type=Path)
    freeze.add_argument("--output", type=Path)

    configure = sub.add_parser("configure-b1")
    configure.add_argument("--frozen", required=True, type=Path)
    configure.add_argument("--base-config", required=True, type=Path)
    configure.add_argument("--output", required=True, type=Path)
    configure.add_argument("--artifact-root", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.action == "prepare":
        _prepare(args)
    elif args.action == "train":
        if not args.confirm_training:
            raise SystemExit("Long final-refit training is blocked; repeat with --confirm-training after reviewing the contract")
        _train(args)
    elif args.action == "freeze":
        _freeze(args)
    else:
        _configure(args)


if __name__ == "__main__":
    main()
