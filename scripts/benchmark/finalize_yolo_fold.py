#!/usr/bin/env python3
"""Infer, select development thresholds and freeze one completed YOLO fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from thesis_fitzpatrick.benchmark import atomic_write_json, load_json  # noqa: E402
from thesis_fitzpatrick.yolo import collect_raw_validation_detections, freeze_detector, select_validation_configuration, validate_completed_training  # noqa: E402


def choose_weights(fold_root: Path, explicit: Path | None) -> Path:
    if explicit:
        if not explicit.is_file(): raise FileNotFoundError(explicit)
        return explicit
    state = validate_completed_training(fold_root / "training" / "training_state.json")
    candidate = Path(state["final_weights_path"])
    if not candidate.is_file():
        raise ValueError(f"Missing validated final Darknet checkpoint: {candidate}")
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-root", required=True, type=Path); parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--manifest", required=True, type=Path); parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--folds", required=True, type=Path); parser.add_argument("--weights", type=Path)
    args = parser.parse_args()
    manifest, folds = load_json(args.manifest), load_json(args.folds)
    if manifest.get("split") != "train": raise SystemExit("YOLO finalization accepts only grouped training folds, never test")
    fold = next((item for item in folds.get("folds", []) if item.get("fold") == args.fold), None)
    if fold is None: raise SystemExit("Unknown fold")
    state_path = args.fold_root / "training" / "training_state.json"
    validate_completed_training(state_path, expected_fold=args.fold)
    cfg = args.fold_root / "lesion-yolov3.cfg"; weights = choose_weights(args.fold_root, args.weights)
    raw_path = args.fold_root / "raw_validation.json"; report_path = args.fold_root / "validation.json"
    records = collect_raw_validation_detections(cfg, weights, manifest, args.data_root, set(fold["validation_ids"]))
    raw_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    report = select_validation_configuration(records, [0.1, 0.2, 0.3, 0.4, 0.5], [0.3, 0.4, 0.5], [0.0, 0.1, 0.2, 0.3])
    atomic_write_json(report_path, report)
    selected = report["selected"]
    frozen = freeze_detector(cfg, weights, {
        "confidence_threshold": selected["confidence_threshold"],
        "nms_threshold": selected["nms_threshold"],
        "margin_fraction": selected["margin_fraction"],
    }, args.fold_root / "frozen.json", report_path, fold=args.fold, training_state=state_path)
    print(json.dumps({"fold": args.fold, "validation_records": len(records), "selected": selected, "frozen": frozen}, indent=2))


if __name__ == "__main__": main()
