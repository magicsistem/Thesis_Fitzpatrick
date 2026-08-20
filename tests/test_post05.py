from __future__ import annotations

from copy import deepcopy

import pytest

from thesis_fitzpatrick.benchmark import content_hash
from thesis_fitzpatrick.post05 import (
    COMPONENT_ABLATIONS,
    build_final_b1_config,
    choose_top3,
    infer_column_map,
    scaled_final_batches,
    validate_final_yolo_manifest,
)


def _frozen() -> dict:
    payload = {
        "schema_version": 3,
        "status": "frozen",
        "architecture": "YOLOv3-Darknet53",
        "model_role": "final_refit",
        "fold": None,
        "cfg_path": "/tmp/cfg",
        "cfg_sha256": "cfg",
        "weights_path": "/tmp/weights",
        "weights_sha256": "weights",
        "training_state_path": "/tmp/state",
        "training_state_sha256": "state",
        "thresholds": {"confidence_threshold": 0.1, "nms_threshold": 0.3, "margin_fraction": 0.0},
    }
    payload["identity_hash"] = content_hash(payload)
    return payload


def test_final_refit_manifest_rejects_cv_fold() -> None:
    payload = _frozen()
    payload.pop("identity_hash")
    payload["model_role"] = "cv_fold"
    payload["fold"] = 0
    payload["identity_hash"] = content_hash({key: value for key, value in payload.items() if key != "identity_hash"})
    with pytest.raises(ValueError, match="final_refit"):
        validate_final_yolo_manifest(payload, verify_artifacts=False)


def test_scaled_final_batches_preserves_cv_exposure() -> None:
    folds = {"folds": [{"fold": i, "train_ids": [f"x{j}" for j in range(80)], "validation_ids": []} for i in range(5)]}
    assert scaled_final_batches(100, folds, 6000) == 7500


def test_build_final_b1_config_records_final_role(monkeypatch: pytest.MonkeyPatch) -> None:
    frozen = _frozen()
    monkeypatch.setattr("thesis_fitzpatrick.post05.validate_final_yolo_manifest", lambda payload, verify_artifacts=True: payload)
    base = {
        "artifact_root": "results/benchmark_v1",
        "p0": {
            "yolo": {"frozen_manifest": "old-fold.json", "cfg_path": None, "weights_path": None, "confidence_threshold": None, "nms_threshold": None},
            "roi": {"margin_fraction": 0.2},
        },
    }
    config = build_final_b1_config(base, frozen)
    assert config["p0"]["yolo"]["model_role"] == "final_refit"
    assert config["p0"]["yolo"]["frozen_manifest"] is None
    assert config["p0"]["roi"]["margin_fraction"] == 0.0
    assert config["post05_contract"]["cv_fold_checkpoint_allowed_for_b1"] is False


def _rows(method: str, dice: float, failure: bool = False, degenerate: bool = False) -> list[dict]:
    return [{
        "method_id": method,
        "condition": "B1",
        "dice": dice,
        "jaccard": max(0.0, dice - 0.05),
        "boundary_f1": max(0.0, dice - 0.08),
        "hd95_normalized": 0.05,
        "technical_failure": failure,
        "degenerate_prediction": degenerate,
    } for _ in range(10)]


def _ablations(method: str, dice: float) -> list[dict]:
    rows = []
    for index, condition in enumerate(COMPONENT_ABLATIONS):
        rows.extend({
            "method_id": method,
            "condition": condition,
            "dice": dice - index * 0.01,
            "technical_failure": False,
            "degenerate_prediction": False,
        } for _ in range(10))
    return rows


def test_top3_score_is_deterministic_and_gated() -> None:
    full = []
    ablation = []
    for method, dice in (("S01", 0.90), ("S02", 0.85), ("S03", 0.80), ("S04", 0.95)):
        full.extend(_rows(method, dice, degenerate=(method == "S04")))
        ablation.extend(_ablations(method, dice))
    ranking = choose_top3(full, ablation, max_technical_failure_rate=0.01, max_degenerate_rate=0.10)
    assert [row["method_id"] for row in ranking if row["eligible"]][:3] == ["S01", "S02", "S03"]
    excluded = next(row for row in ranking if row["method_id"] == "S04")
    assert excluded["eligible"] is False


def test_column_aliases_detect_mst_and_lstar() -> None:
    mapping = infer_column_map(["image_name", "Monk Skin Tone", "L*", "Fitzpatrick"])
    # image_name is intentionally not silently treated as image_id; dataset-specific
    # names must be mapped explicitly, while tone/color aliases are recognized.
    assert mapping["mst"] == "Monk Skin Tone"
    assert mapping["l_star"] == "L*"
    assert mapping["fst"] == "Fitzpatrick"
