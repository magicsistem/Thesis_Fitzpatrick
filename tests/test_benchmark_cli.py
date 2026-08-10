from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent


class BenchmarkCliTests(unittest.TestCase):
    def test_s16_native_manifest_run_exports_prediction_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "masks").mkdir()
            rgb = np.full((72, 96, 3), (180, 140, 115), dtype=np.uint8)
            cv2.ellipse(rgb, (48, 36), (18, 13), 0, 0, 360, (45, 25, 20), -1)
            gt = np.zeros((72, 96), dtype=np.uint8)
            cv2.ellipse(gt, (48, 36), (18, 13), 0, 0, 360, 255, -1)
            Image.fromarray(rgb).save(root / "images" / "one.png")
            Image.fromarray(gt).save(root / "masks" / "one.png")
            manifest = {
                "schema_version": 1,
                "dataset_id": "synthetic_test",
                "split": "validation",
                "items": [{
                    "image_id": "one",
                    "image_path": "images/one.png",
                    "mask_paths": ["masks/one.png"],
                    "patient_id": None,
                    "lesion_id": None,
                    "duplicate_group_id": "one",
                }],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            process = subprocess.run(
                [
                    sys.executable, "scripts/benchmark/run_evaluation.py",
                    "--evaluation", "A", "--dataset", "synthetic_test", "--split", "validation",
                    "--models", "S16", "--manifest", str(manifest_path), "--data-root", str(root),
                    "--artifact-root", str(root / "artifacts"), "--run-id", "integration-a",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            result_path = root / "artifacts" / "runs" / "integration-a" / "predictions" / "S16" / "one" / "result.json"
            result = json.loads(result_path.read_text())
            self.assertEqual(result["evaluation"], "A")
            self.assertIsNotNone(result["metrics"])
            self.assertTrue((result_path.parent / "native_mask.png").is_file())
            self.assertTrue((result_path.parent / "final_mask.png").is_file())
            self.assertTrue((result_path.parent / "overlay.jpg").is_file())
            self.assertTrue((root / "artifacts" / "runs" / "integration-a" / "inputs" / "one.ground_truth.png").is_file())
            run = json.loads((root / "artifacts" / "runs" / "integration-a" / "run_manifest.json").read_text())
            self.assertEqual(run["status"], "completed")

            common = subprocess.run(
                [
                    sys.executable, "scripts/benchmark/run_evaluation.py",
                    "--evaluation", "B1", "--dataset", "synthetic_test", "--split", "validation",
                    "--models", "S16", "--manifest", str(manifest_path), "--data-root", str(root),
                    "--artifact-root", str(root / "artifacts"), "--run-id", "integration-b1",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(common.returncode, 0, common.stderr)
            common_result = json.loads(
                (root / "artifacts" / "runs" / "integration-b1" / "predictions" / "S16" / "one" / "result.json").read_text()
            )
            self.assertEqual(common_result["evaluation"], "B1")
            self.assertFalse(common_result["p0_cache_hit"])
            self.assertTrue((root / "artifacts" / "preprocessing" / "synthetic_test" / "one").is_dir())
            self.assertEqual(len(list((root / "artifacts" / "preprocessing" / "synthetic_test" / "one").glob("*/yolo_overlay.png"))), 1)

    def test_b2_refuses_to_reuse_b1_checkpoints(self) -> None:
        process = subprocess.run(
            [
                sys.executable, "scripts/benchmark/run_evaluation.py",
                "--evaluation", "B2", "--dataset", "synthetic_test", "--split", "validation",
                "--models", "S16", "--image", "README.md", "--dry-run",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("faltan checkpoints B2", process.stderr)

    def test_ablation_dry_run_registers_all_sixteen_backends(self) -> None:
        process = subprocess.run(
            [sys.executable, "scripts/benchmark/run_ablation.py", "--dataset", "synthetic_test", "--split", "validation", "--models", "all", "--image", "README.md", "--dry-run"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        payload = json.loads(process.stdout)
        self.assertEqual(len(payload["methods"]), 16)
        self.assertEqual(payload["executions"], 64)


if __name__ == "__main__":
    unittest.main()
