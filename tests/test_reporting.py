from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from thesis_fitzpatrick.reporting import aggregate, load_run_rows  # noqa: E402


class ReportingOutcomeTests(unittest.TestCase):
    def test_empty_prediction_counts_scientifically_but_adapter_error_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            cases = [
                (
                    "empty",
                    {
                        "schema_version": 1,
                        "evaluation": "B1",
                        "image_id": "empty",
                        "method_id": "S16",
                        "failure_code": "empty_mask",
                        "backend": {"failure_code": "empty_mask", "backend_time_ms": 1.0},
                        "metadata": {},
                        "metrics": {"jaccard": 0.0, "threshold_jaccard": 0.0, "dice": 0.0, "flags": {"prediction_empty": True}},
                    },
                ),
                (
                    "technical",
                    {
                        "schema_version": 1,
                        "evaluation": "B1",
                        "image_id": "technical",
                        "method_id": "S16",
                        "failure_code": "adapter_error",
                        "backend": {"failure_code": "adapter_error", "backend_time_ms": 1.0},
                        "metadata": {},
                        "metrics": {"jaccard": 0.0, "threshold_jaccard": 0.0, "dice": 0.0, "flags": {"prediction_empty": True}},
                    },
                ),
            ]
            for image_id, payload in cases:
                path = run / "predictions" / "S16" / image_id / "result.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(payload), encoding="utf-8")

            rows = load_run_rows(run)
            summary = aggregate(rows, repetitions=20, seed=3)[0]
            self.assertEqual(summary["n"], 2)
            self.assertEqual(summary["scientific_n"], 1)
            self.assertEqual(summary["technical_failures"], 1)
            self.assertEqual(summary["degenerate_predictions"], 1)
            self.assertEqual(summary["prediction_empty"], 2)
            self.assertEqual(summary["jaccard_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
