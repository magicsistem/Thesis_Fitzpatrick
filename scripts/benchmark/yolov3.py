#!/usr/bin/env python3
"""Prepare, train, resume, and freeze one-class YOLOv3/Darknet-53."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis_fitzpatrick.benchmark import atomic_write_json, load_json  # noqa: E402
from thesis_fitzpatrick.yolo import DarknetInterrupted, collect_raw_validation_detections, discover_darknet_resume, freeze_detector, patch_yolov3_cfg, prepare_darknet_fold, run_darknet_training, select_validation_configuration, validate_frozen_yolo, validate_frozen_yolo_set  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare-fold")
    prepare.add_argument("--manifest", required=True, type=Path); prepare.add_argument("--data-root", required=True, type=Path)
    prepare.add_argument("--folds", required=True, type=Path); prepare.add_argument("--fold", required=True, type=int)
    prepare.add_argument("--output", required=True, type=Path); prepare.add_argument("--seed", type=int, default=20260806)
    prepare.add_argument("--box-perturbation", type=float, default=0.05)
    cfg = sub.add_parser("configure")
    cfg.add_argument("--source-cfg", required=True, type=Path); cfg.add_argument("--output", required=True, type=Path)
    cfg.add_argument("--batch", type=int, default=64); cfg.add_argument("--subdivisions", type=int, default=16); cfg.add_argument("--input-size", type=int, default=512)
    cfg.add_argument("--learning-rate", type=float, default=0.001); cfg.add_argument("--momentum", type=float, default=0.9); cfg.add_argument("--weight-decay", type=float, default=0.0005)
    cfg.add_argument("--max-batches", type=int, default=6000)
    train = sub.add_parser("train")
    train.add_argument("--darknet", required=True, type=Path); train.add_argument("--data", required=True, type=Path)
    train.add_argument("--cfg", required=True, type=Path); train.add_argument("--initial-weights", required=True, type=Path)
    train.add_argument("--fold", required=True, type=int); train.add_argument("--resume", type=Path); train.add_argument("--output", required=True, type=Path); train.add_argument("--confirm-training", action="store_true")
    resume_plan = sub.add_parser("resume-plan", help="Read-only checkpoint compatibility report")
    resume_plan.add_argument("--darknet", required=True, type=Path); resume_plan.add_argument("--data", required=True, type=Path)
    resume_plan.add_argument("--cfg", required=True, type=Path); resume_plan.add_argument("--initial-weights", required=True, type=Path)
    resume_plan.add_argument("--fold", required=True, type=int); resume_plan.add_argument("--output", required=True, type=Path)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--cfg", required=True, type=Path); freeze.add_argument("--weights", required=True, type=Path)
    freeze.add_argument("--validation-report", required=True, type=Path); freeze.add_argument("--confidence", required=True, type=float)
    freeze.add_argument("--nms", required=True, type=float); freeze.add_argument("--margin", required=True, type=float); freeze.add_argument("--fold", required=True, type=int); freeze.add_argument("--training-state", required=True, type=Path); freeze.add_argument("--output", required=True, type=Path)
    validate = sub.add_parser("validate", help="Select confidence/NMS/margin only from recorded fold validation detections")
    validate.add_argument("--predictions", required=True, type=Path); validate.add_argument("--output", required=True, type=Path)
    validate.add_argument("--confidence-candidates", default="0.1,0.2,0.3,0.4,0.5"); validate.add_argument("--nms-candidates", default="0.3,0.4,0.5"); validate.add_argument("--margin-candidates", default="0,0.1,0.2,0.3")
    infer = sub.add_parser("infer-validation", help="Run raw detections only for a grouped fold's validation IDs")
    infer.add_argument("--cfg", required=True, type=Path); infer.add_argument("--weights", required=True, type=Path); infer.add_argument("--manifest", required=True, type=Path); infer.add_argument("--data-root", required=True, type=Path); infer.add_argument("--folds", required=True, type=Path); infer.add_argument("--fold", required=True, type=int); infer.add_argument("--output", required=True, type=Path)
    gate = sub.add_parser("validate-folds", help="Require five complete, frozen and hash-verified YOLO folds")
    gate.add_argument("--root", required=True, type=Path)
    frozen_gate = sub.add_parser("validate-frozen", help="Validate one frozen YOLO fold")
    frozen_gate.add_argument("--manifest", required=True, type=Path); frozen_gate.add_argument("--fold", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "prepare-fold":
        manifest, folds = load_json(args.manifest), load_json(args.folds)
        if manifest.get("split") != "train":
            raise SystemExit("Las etiquetas YOLO solo se generan desde el split train.")
        fold = next((item for item in folds["folds"] if item["fold"] == args.fold), None)
        if fold is None: raise SystemExit("Fold inexistente")
        print(json.dumps(prepare_darknet_fold(manifest, args.data_root, fold, args.output, seed=args.seed, perturbation_fraction=args.box_perturbation), indent=2))
    elif args.action == "configure":
        patch_yolov3_cfg(args.source_cfg, args.output, batch=args.batch, subdivisions=args.subdivisions, width=args.input_size, height=args.input_size, learning_rate=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, max_batches=args.max_batches)
        print(args.output)
    elif args.action == "train":
        if not args.confirm_training:
            raise SystemExit("Entrenamiento largo bloqueado. Revise recursos/comando y repita con --confirm-training.")
        try:
            print(json.dumps(run_darknet_training(args.darknet, args.data, args.cfg, args.initial_weights, args.output, fold=args.fold, resume_weights=args.resume), indent=2))
        except DarknetInterrupted as exc:
            print(f"YOLO interrupted and checkpoint inventory was recorded: {exc}", file=sys.stderr)
            raise SystemExit(75) from exc
    elif args.action == "resume-plan":
        print(json.dumps(discover_darknet_resume(args.darknet, args.data, args.cfg, args.initial_weights, args.output, fold=args.fold), indent=2))
    elif args.action == "validate":
        records = json.loads(args.predictions.read_text(encoding="utf-8"))
        report = select_validation_configuration(records, *[[float(value) for value in getattr(args, name).split(",")] for name in ("confidence_candidates", "nms_candidates", "margin_candidates")])
        atomic_write_json(args.output, report); print(json.dumps(report["selected"], indent=2))
    elif args.action == "infer-validation":
        folds = load_json(args.folds); fold = next((item for item in folds["folds"] if item["fold"] == args.fold), None)
        if fold is None: raise SystemExit("Fold inexistente")
        records = collect_raw_validation_detections(args.cfg, args.weights, load_json(args.manifest), args.data_root, set(fold["validation_ids"]))
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8"); print(json.dumps({"records": len(records), "output": str(args.output)}, indent=2))
    elif args.action == "validate-folds":
        print(json.dumps({"status": "valid", "folds": [item["fold"] for item in validate_frozen_yolo_set(args.root)]}, indent=2))
    elif args.action == "validate-frozen":
        print(json.dumps(validate_frozen_yolo(args.manifest, args.fold), indent=2))
    else:
        print(json.dumps(freeze_detector(args.cfg, args.weights, {"confidence_threshold": args.confidence, "nms_threshold": args.nms, "margin_fraction": args.margin}, args.output, args.validation_report, fold=args.fold, training_state=args.training_state), indent=2))


if __name__ == "__main__":
    main()
