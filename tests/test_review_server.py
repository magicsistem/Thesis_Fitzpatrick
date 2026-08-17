from __future__ import annotations

import json
import tempfile
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

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
        self.assertEqual(len(payload["benchmark"]["methods"]), 16)
        evaluations = {item["id"]: item for item in payload["benchmark"]["evaluations"]}
        self.assertTrue(evaluations["B1"]["available"])
        self.assertFalse(evaluations["B2"]["available"])
        self.assertFalse(payload["benchmark"]["resources"]["yolov3"]["available"])

    def test_benchmark_state_endpoint_exposes_locked_resources(self) -> None:
        with urlopen(f"{self.base_url}/api/benchmark/state") as response:
            payload = json.load(response)
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["implementation"]["s16_classic"], "ready")
        self.assertEqual(payload["implementation"]["sealed_execution"], "ready_and_locked")

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
        self.assertIn("id=\"select-all-models\"", page)
        self.assertIn("id=\"native-run\"", page)
        self.assertIn("id=\"native-run-view\"", page)
        app = (REPO_ROOT / "web" / "app.js").read_text(encoding="utf-8")
        self.assertIn('run.evaluation === "A"', app)
        self.assertIn('["B1", "B2"].includes(run.evaluation)', app)
        self.assertIn("Fallos técnicos", app)
        self.assertIn("degenerate_predictions", app)
        self.assertIn("failure_is_fatal", app)


    def test_benchmark_run_exposes_clean_skin_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction = root / "runs" / "native-a" / "predictions" / "S16" / "one"
            prediction.mkdir(parents=True)
            (root / "runs" / "native-a" / "run_manifest.json").write_text(json.dumps({"dataset": "synthetic", "split": "validation"}), encoding="utf-8")
            (prediction / "result.json").write_text(json.dumps({
                "image_id": "one", "dataset_id": "synthetic", "evaluation": "A",
                "method_id": "S16", "clean_skin_mask": "clean_skin_mask.png",
            }), encoding="utf-8")
            Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(prediction / "clean_skin_mask.png")
            with patch.object(review, "BENCHMARK_ARTIFACT_ROOT", root):
                payload = review.benchmark_run("native-a")
            self.assertEqual(
                payload["results"][0]["artifacts"]["clean_skin_mask.png"],
                "/files/benchmark/native-a/predictions/S16/one/clean_skin_mask.png",
            )

    def test_benchmark_run_distinguishes_degenerate_from_technical_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs" / "outcomes"
            (run / "predictions" / "S16" / "empty").mkdir(parents=True)
            (run / "predictions" / "S03" / "killed").mkdir(parents=True)
            (run / "run_manifest.json").write_text(json.dumps({"dataset": "synthetic", "split": "validation"}), encoding="utf-8")
            (run / "predictions" / "S16" / "empty" / "result.json").write_text(json.dumps({
                "image_id": "empty", "dataset_id": "synthetic", "evaluation": "B1",
                "method_id": "S16", "failure_code": "empty_mask",
                "backend": {"failure_code": "empty_mask"},
                "metrics": {"flags": {"prediction_empty": True}},
            }), encoding="utf-8")
            (run / "predictions" / "S03" / "killed" / "result.json").write_text(json.dumps({
                "image_id": "killed", "dataset_id": "synthetic", "evaluation": "A",
                "method_id": "S03", "failure_code": "adapter_error",
                "backend": {"failure_code": "adapter_error"},
                "metrics": {"flags": {"prediction_empty": True}},
            }), encoding="utf-8")
            with patch.object(review, "BENCHMARK_ARTIFACT_ROOT", root):
                payload = review.benchmark_run("outcomes")
            by_method = {item["method_id"]: item for item in payload["results"]}
            self.assertFalse(by_method["S16"]["failure_is_fatal"])
            self.assertEqual(by_method["S16"]["outcome"], "degenerate_prediction")
            self.assertTrue(by_method["S03"]["failure_is_fatal"])
            self.assertEqual(by_method["S03"]["outcome"], "technical_failure")

    def test_inference_accepts_every_catalogued_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "input.jpg"
            Image.fromarray(np.full((32, 32, 3), [120, 85, 65], dtype=np.uint8)).save(image_path)
            results = root / "results"
            model_ids = [model["id"] for model in review.load_models()]
            for model_id in model_ids:
                lesion_path = results / model_id / "image" / "lesion_mask.png"
                lesion_path.parent.mkdir(parents=True)
                lesion = np.zeros((32, 32), dtype=np.uint8)
                lesion[12:20, 12:20] = 255
                Image.fromarray(lesion).save(lesion_path)
            image_record = {"id": "image", "url": "/files/input/input.jpg"}
            request = Request(
                f"{self.base_url}/api/infer",
                data=json.dumps({"model_ids": model_ids, "image_ids": ["image"]}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with (
                patch.object(review, "RESULTS_DIR", results),
                patch.object(review, "list_images", return_value=[image_record]),
                patch.object(review, "image_path", return_value=image_path),
                urlopen(request) as response,
            ):
                payload = json.load(response)
            self.assertEqual(len(payload["results"]), len(model_ids))
            self.assertTrue(all(item["status"] == "ready" for item in payload["results"]))

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
            with patch.object(review, "REPO_ROOT", root), patch.dict(
                review.os.environ,
                {"THESIS_ADAPTER_PYTHON": sys.executable},
            ):
                success, _, _ = review.run_adapter(model, Path("unused.jpg"), lesion_path)
            self.assertTrue(success)


if __name__ == "__main__":
    unittest.main()
