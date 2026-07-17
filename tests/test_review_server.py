from __future__ import annotations

import json
import tempfile
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch
from urllib.request import urlopen

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import serve_segmentation_review as review

ReviewHandler = review.ReviewHandler
ThreadingHTTPServer = review.ThreadingHTTPServer


class ReviewServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), ReviewHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_state_lists_fifteen_runnable_checkpoint_variants(self) -> None:
        with urlopen(f"{self.base_url}/api/state") as response:
            payload = json.load(response)
        self.assertEqual(len(payload["models"]), 15)
        self.assertEqual(sum(model["recommended"] for model in payload["models"]), 10)
        for model in payload["models"]:
            self.assertTrue(model["year"])
            self.assertTrue(model["license"])
            self.assertTrue(model["repository"].startswith("https://"))
            self.assertTrue(model["description"])
            self.assertTrue(model["resource_profile"])
            self.assertIsNotNone(model["adapter_command"])
            self.assertIn("verified", model["checkpoint_status"])
        self.assertIn("pool", payload)

    def test_stratified_sample_balances_six_fitzpatrick_types(self) -> None:
        images = [
            {"id": f"{fitzpatrick_type}-{index}", "fitzpatrick_skin_type": fitzpatrick_type}
            for fitzpatrick_type in review.FITZPATRICK_TYPES
            for index in range(4)
        ]
        first = review.stratified_sample(images, total=12, seed=17)
        second = review.stratified_sample(images, total=12, seed=17)
        self.assertEqual(first, second)
        for fitzpatrick_type in review.FITZPATRICK_TYPES:
            self.assertEqual(
                sum(item["fitzpatrick_skin_type"] == fitzpatrick_type for item in first),
                2,
            )

    def test_stratified_sample_requires_multiple_of_six(self) -> None:
        with self.assertRaisesRegex(ValueError, "múltiplo de 6"):
            review.stratified_sample([], total=10)

    def test_index_is_served(self) -> None:
        with urlopen(f"{self.base_url}/") as response:
            page = response.read().decode("utf-8")
        self.assertIn("Comparación de máscaras", page)
        self.assertIn("id=\"carousel\"", page)

    def test_ready_result_writes_skin_mask_and_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "input.jpg"
            Image.fromarray(np.full((48, 64, 3), [120, 85, 65], dtype=np.uint8)).save(image_path)
            results = root / "results"
            lesion_path = results / "model" / "image" / "lesion_mask.png"
            lesion_path.parent.mkdir(parents=True)
            lesion = np.zeros((48, 64), dtype=np.uint8)
            lesion[16:32, 24:40] = 255
            Image.fromarray(lesion).save(lesion_path)

            with patch.object(review, "RESULTS_DIR", results):
                payload = review.result_payload("model", "image", image_path)

            self.assertEqual(payload["status"], "ready")
            self.assertTrue((lesion_path.parent / "clean_skin_mask.png").is_file())
            self.assertTrue((lesion_path.parent / "colour_stats.json").is_file())

    def test_full_lesion_mask_keeps_result_ready_without_skin_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "input.jpg"
            Image.fromarray(np.full((32, 32, 3), [120, 85, 65], dtype=np.uint8)).save(image_path)
            results = root / "results"
            lesion_path = results / "model" / "image" / "lesion_mask.png"
            lesion_path.parent.mkdir(parents=True)
            Image.fromarray(np.full((32, 32), 255, dtype=np.uint8)).save(lesion_path)

            with patch.object(review, "RESULTS_DIR", results):
                payload = review.result_payload("model", "image", image_path)

            self.assertEqual(payload["status"], "ready")
            self.assertIsNone(payload["stats"]["skin_colour"])
            self.assertFalse(payload["stats"]["skin_colour_available"])
            self.assertIn("no dejó píxeles", payload["stats"]["skin_colour_error"])
            self.assertTrue((lesion_path.parent / "clean_skin_mask.png").is_file())

    def test_adapter_collects_runtime_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lesion_path = Path(directory) / "lesion_mask.png"
            model = {
                "adapter_command": [
                    sys.executable,
                    "-c",
                    "from PIL import Image; import sys; Image.new('L', (2, 2), 255).save(sys.argv[1])",
                    "{lesion_mask}",
                ]
            }
            success, _, metrics = review.run_adapter(model, Path("unused.jpg"), lesion_path)

            self.assertTrue(success)
            self.assertIsNotNone(metrics)
            self.assertIn("wall_seconds", metrics)
            self.assertIn("effective_cpu_cores", metrics)
            self.assertIn("cpu_percent_single_core_equivalent", metrics)
            self.assertIn("peak_ram_mb", metrics)

    def test_benchmark_summary_averages_per_model(self) -> None:
        runtime = {
            "wall_seconds": 2.0,
            "cpu_percent_single_core_equivalent": 50.0,
            "cpu_percent_system_capacity": 5.0,
            "peak_ram_mb": 100.0,
        }
        summaries = review.benchmark_summaries([
            {"model_id": "a", "status": "ready", "runtime": runtime},
            {"model_id": "a", "status": "ready", "runtime": runtime},
        ])
        self.assertEqual(summaries[0]["images_timed"], 2)
        self.assertEqual(summaries[0]["average_wall_seconds"], 2.0)
        self.assertEqual(summaries[0]["average_effective_cpu_cores"], 0.5)
        self.assertEqual(summaries[0]["peak_ram_mb_max"], 100.0)

    def test_adapter_expands_placeholders_inside_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lesion_path = root / "lesion_mask.png"
            model = {
                "adapter_command": [
                    "{python}",
                    "-c",
                    "from PIL import Image; import sys; Image.new('L', (2, 2), 255).save(sys.argv[1])",
                    "{repo_root}/lesion_mask.png",
                ]
            }
            with patch.object(review, "REPO_ROOT", root):
                success, _, _ = review.run_adapter(model, Path("unused.jpg"), lesion_path)
            self.assertTrue(success)


if __name__ == "__main__":
    unittest.main()
