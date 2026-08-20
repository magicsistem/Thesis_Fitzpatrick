#!/usr/bin/env python3
"""Dependency-light post-0.5 contract self-check for CEDIA preflight."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis_fitzpatrick.benchmark import content_hash  # noqa: E402
from thesis_fitzpatrick.post05 import (  # noqa: E402
    COMPONENT_ABLATIONS,
    choose_top3,
    scaled_final_batches,
    validate_final_yolo_manifest,
)


def main() -> None:
    folds = {"folds": [{"fold": i, "train_ids": [f"x{j}" for j in range(80)], "validation_ids": []} for i in range(5)]}
    assert scaled_final_batches(100, folds, 6000) == 7500

    frozen = {
        "schema_version": 3,
        "status": "frozen",
        "architecture": "YOLOv3-Darknet53",
        "model_role": "final_refit",
        "fold": None,
        "cfg_path": "/not/checked/cfg",
        "cfg_sha256": "x",
        "weights_path": "/not/checked/weights",
        "weights_sha256": "y",
        "training_state_path": "/not/checked/state",
        "training_state_sha256": "z",
        "thresholds": {"confidence_threshold": 0.1, "nms_threshold": 0.3, "margin_fraction": 0.0},
    }
    frozen["identity_hash"] = content_hash(frozen)
    validate_final_yolo_manifest(frozen, verify_artifacts=False)

    full = []
    ablation = []
    for method, dice in (("S01", 0.90), ("S02", 0.85), ("S03", 0.80)):
        for _ in range(5):
            full.append({
                "method_id": method,
                "condition": "B1",
                "dice": dice,
                "jaccard": dice - 0.05,
                "boundary_f1": dice - 0.08,
                "hd95_normalized": 0.05,
                "technical_failure": False,
                "degenerate_prediction": False,
            })
        for index, condition in enumerate(COMPONENT_ABLATIONS):
            for _ in range(5):
                ablation.append({
                    "method_id": method,
                    "condition": condition,
                    "dice": dice - index * 0.01,
                    "technical_failure": False,
                    "degenerate_prediction": False,
                })
    ranked = choose_top3(full, ablation)
    assert [row["method_id"] for row in ranked[:3]] == ["S01", "S02", "S03"]
    print("post05 self-check OK")


if __name__ == "__main__":
    main()
