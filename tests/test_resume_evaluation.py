from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "benchmark"))

import resume_evaluation  # noqa: E402


class ResumeEvaluationTests(unittest.TestCase):
    def test_scan_keeps_empty_mask_and_rejects_adapter_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest = {
                "evaluation": "B1",
                "dataset": "synthetic",
                "split": "validation",
                "images": ["one"],
                "methods": ["S01", "S16"],
            }
            payloads = {
                "S01": {
                    "schema_version": 1,
                    "evaluation": "B1",
                    "dataset_id": "synthetic",
                    "split": "validation",
                    "image_id": "one",
                    "method_id": "S01",
                    "failure_code": "adapter_error",
                    "backend": {"failure_code": "adapter_error", "warnings": ["killed"]},
                    "metrics": {"jaccard": 0.0, "flags": {"prediction_empty": True}},
                    "p0_fallback_used": False,
                },
                "S16": {
                    "schema_version": 1,
                    "evaluation": "B1",
                    "dataset_id": "synthetic",
                    "split": "validation",
                    "image_id": "one",
                    "method_id": "S16",
                    "failure_code": "empty_mask",
                    "backend": {"failure_code": "empty_mask", "warnings": ["empty"]},
                    "metrics": {
                        "threshold_jaccard": 0.0,
                        "jaccard": 0.0,
                        "dice": 0.0,
                        "flags": {"prediction_empty": True},
                    },
                    "p0_fallback_used": True,
                },
            }
            for method_id, payload in payloads.items():
                prediction = run / "predictions" / method_id / "one"
                prediction.mkdir(parents=True)
                (prediction / "result.json").write_text(json.dumps(payload), encoding="utf-8")
                for name in ("native_mask.png", "final_mask.png", "clean_skin_mask.png"):
                    (prediction / name).write_bytes(b"x")

            result = resume_evaluation.scan_run(run, manifest, store_web_previews=False)
            self.assertEqual(result["counts"]["expected"], 2)
            self.assertEqual(result["counts"]["technical_failures"], 1)
            self.assertEqual(result["counts"]["scientific_metrics"], 1)
            self.assertEqual(result["counts"]["quality_flagged_predictions"], 1)
            self.assertEqual(result["technical_failures"][0]["method_id"], "S01")
            self.assertEqual(result["quality_flags"][0]["method_id"], "S16")
            self.assertEqual(result["quality_flags"][0]["quality_flags"], "empty_mask | prediction_empty")


if __name__ == "__main__":
    unittest.main()
