from __future__ import annotations

import json
import io
from pathlib import Path
import sys
import tempfile
import unittest
import subprocess

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "benchmark"))

from thesis_fitzpatrick.annotations import export_consensus_manifest, initialize_project, project_status, save_mask_version
from thesis_fitzpatrick.benchmark import atomic_write_json
from thesis_fitzpatrick.datasets import OFFICIAL_IMAPP_FILES, audit_manifest_overlap, build_isic2018_manifest, build_novice_manifest, download_resumable, file_digest, import_imapp_metadata, majority_consensus, staple_consensus, verify_manifest_files
from thesis_fitzpatrick.reporting import aggregate, paired_comparisons, write_report
from thesis_fitzpatrick.yolo import bbox_from_mask, bbox_to_darknet, darknet_to_bbox, patch_yolov3_cfg, prepare_darknet_fold, select_validation_configuration
from thesis_fitzpatrick.benchmark import content_hash, sha256_file
import sealed_test
from setup_yolov3_darknet import cpu_build_command, gpu_build_command


class DatasetAndYoloTests(unittest.TestCase):
    def test_isic_import_never_mixes_official_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, image_id in (("Training", "ISIC_train"), ("Validation", "ISIC_validation")):
                image_dir = root / f"ISIC2018_Task1-2_{split}_Input"; mask_dir = root / f"ISIC2018_Task1_{split}_GroundTruth"; image_dir.mkdir(); mask_dir.mkdir()
                Image.new("RGB", (8, 6), "white").save(image_dir / f"{image_id}.jpg"); Image.new("L", (8, 6), 255).save(mask_dir / f"{image_id}_segmentation.png")
            metadata = root / "metadata"; metadata.mkdir(); (metadata / "ISIC_train.json").write_text(json.dumps({"copyright_license":"CC-0", "metadata":{"clinical":{"patient_id":"patient", "lesion_id":"lesion"}}}), encoding="utf-8")
            result = build_isic2018_manifest(root, "train", include_checksums=False, metadata_root=metadata)
            self.assertEqual([item["image_id"] for item in result["items"]], ["ISIC_train"])
            self.assertIn("Training_GroundTruth", result["items"][0]["mask_paths"][0])
            self.assertEqual(result["items"][0]["duplicate_group_id"], "lesion")

    def test_darknet_cpu_build_disables_upstream_gpu_defaults(self):
        command = cpu_build_command(2)
        self.assertIn("GPU=0", command); self.assertIn("CUDNN=0", command); self.assertIn("OPENCV=0", command); self.assertIn("-j2", command)

    def test_darknet_gpu_build_targets_one_a100(self):
        command = gpu_build_command(8)
        self.assertIn("GPU=1", command); self.assertIn("CUDNN=0", command)
        self.assertTrue(any("arch=compute_80,code=[sm_80,compute_80]" in token for token in command))
        self.assertNotIn("-gpus", command)

    def test_resumable_download_identifies_client_for_official_hosts(self):
        class Response(io.BytesIO):
            status = 200
            def __enter__(self): return self
            def __exit__(self, *_): self.close()
        with tempfile.TemporaryDirectory() as directory:
            import unittest.mock
            with unittest.mock.patch("urllib.request.urlopen", side_effect=lambda request, timeout: (self.assertIn("Thesis-Fitzpatrick", request.get_header("User-agent")) or self.assertEqual(timeout, 120.0) or Response(b"official"))):
                result = download_resumable("https://official.invalid/file", Path(directory) / "file")
            self.assertEqual(result["bytes"], 8)

    def test_resumable_download_never_overwrites_published_file(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "file"; destination.write_bytes(b"keep")
            import unittest.mock
            with unittest.mock.patch("urllib.request.urlopen", side_effect=AssertionError("network must not be used")):
                result = download_resumable("https://official.invalid/file", destination)
            self.assertTrue(result["reused"]); self.assertEqual(destination.read_bytes(), b"keep")

    def test_novice_mapping_removes_unet_suffix_and_stays_auxiliary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "images").mkdir(); (root / "Masks").mkdir()
            Image.new("RGB", (8, 6), "white").save(root / "images" / "ISIC_1.jpg"); Image.new("L", (8, 6), 255).save(root / "Masks" / "opaque_UNET.png")
            (root / "mapping.csv").write_text("Filename,ISIC_ID\nopaque_UNET.png,ISIC_1\n", encoding="utf-8")
            result = build_novice_manifest(root / "Masks", root, mapping_csv=root / "mapping.csv")
            self.assertEqual(result["items"][0]["image_id"], "ISIC_1"); self.assertFalse(result["enabled"]); self.assertIn("NO APTO", result["warning"])

    def test_imapp_import_normalizes_filename_ids_and_uses_consensus_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "source"; data = root / "data"; source.mkdir(); (data / "images").mkdir(parents=True); (data / "segmentations").mkdir()
            Image.new("RGB", (8, 6), "white").save(data / "images" / "ISIC_0000001.jpg")
            masks = []
            for role in ("A01", "MV", "ST"):
                path = data / "segmentations" / f"ISIC_0000001_{role}.png"; Image.new("L", (8, 6), 255).save(path); masks.append((role, path))
            def write_csv(name, fields, rows):
                import csv
                with (source / name).open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
            write_csv("img_metadata.csv", ["isic_id", "patient_id", "lesion_id", "fitzpatrick_skin_type"], [{"isic_id":"ISIC_0000001", "patient_id":"p", "lesion_id":"l", "fitzpatrick_skin_type":"VI"}])
            rows = [{"ISIC_id":"ISIC_0000001", "seg_filename":path.name, "annotator":role, "mask_md5":file_digest(path, "md5")} for role, path in masks]
            write_csv("seg_metadata.csv", ["ISIC_id", "seg_filename", "annotator", "mask_md5"], rows)
            for name in ("train.csv", "val.csv", "test.csv"): write_csv(name, ["image"], [{"image":"ISIC_0000001.JPG"}])
            write_csv("iaa_metrics_image.csv", ["x"], [{"x":"1"}]); write_csv("iaa_metrics_pairwise.csv", ["x"], [{"x":"1"}])
            verified = dict(OFFICIAL_IMAPP_FILES)
            try:
                for name in verified:
                    if not (source / name).exists(): (source / name).write_bytes(b"")
                import unittest.mock
                with unittest.mock.patch("thesis_fitzpatrick.datasets.OFFICIAL_IMAPP_FILES", {name:file_digest(source / name, "md5") for name in verified}): result = import_imapp_metadata(source, data)
            finally: pass
            item = result["manifests"]["train"]["items"][0]
            self.assertEqual(item["image_id"], "ISIC_0000001"); self.assertIn("_ST.png", item["mask_paths"][0]); self.assertIn("_MV.png", item["mask_paths"][1]); self.assertEqual(item["fitzpatrick"], "VI")

    def test_manifest_integrity_detects_modified_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); Image.new("RGB", (8, 6), "white").save(root / "a.png")
            payload = {"schema_version": 1, "dataset_id": "x", "split": "train", "items": [{"image_id": "a", "image_path": "a.png", "mask_paths": [], "image_sha256": "0" * 64}]}
            report = verify_manifest_files(payload, root)
            self.assertFalse(report["passed"]); self.assertIn("SHA-256", report["corrupt_or_modified"][0]["error"])

    def test_manifest_integrity_verifies_mask_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); Image.new("RGB", (8, 6), "white").save(root / "a.png"); Image.new("L", (8, 6), 255).save(root / "m.png")
            payload = {"schema_version": 1, "dataset_id": "x", "split": "train", "items": [{"image_id": "a", "image_path": "a.png", "mask_paths": ["m.png"], "mask_sha256": "0" * 64}]}
            report = verify_manifest_files(payload, root)
            self.assertFalse(report["passed"]); self.assertIn("máscara", report["corrupt_or_modified"][0]["error"])

    def test_staple_and_majority_are_binary(self):
        masks = [np.pad(np.ones((2, 2), np.uint8), 1), np.pad(np.ones((2, 2), np.uint8), 1), np.zeros((4, 4), np.uint8)]
        np.testing.assert_array_equal(majority_consensus(masks), masks[0])
        self.assertEqual(set(np.unique(staple_consensus(masks))), {0, 1})

    def test_darknet_roundtrip_and_fold_never_uses_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "images").mkdir(); (root / "masks").mkdir()
            items = []
            for image_id in ("a", "b"):
                Image.new("RGB", (20, 10), "white").save(root / "images" / f"{image_id}.jpg")
                mask = np.zeros((10, 20), np.uint8); mask[2:8, 4:15] = 255; cv2.imwrite(str(root / "masks" / f"{image_id}.png"), mask)
                items.append({"image_id": image_id, "image_path": f"images/{image_id}.jpg", "mask_paths": [f"masks/{image_id}.png"]})
            box = bbox_from_mask(mask); self.assertEqual(box.as_list(), darknet_to_bbox(bbox_to_darknet(box, 20, 10), 20, 10).as_list())
            audit = prepare_darknet_fold({"items": items}, root, {"fold": 0, "train_ids": ["a"], "validation_ids": ["b"]}, root / "fold", seed=1)
            self.assertEqual(audit["training_count"], 1); self.assertEqual(audit["validation_count"], 1)

    def test_yolov3_cfg_replaces_coco_schedule_for_one_class(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output = root / "lesion.cfg"; source = root / "yolov3.cfg"
            head = "[net]\nbatch=1\nsubdivisions=1\nwidth=416\nheight=416\nlearning_rate=0.001\nmomentum=0.9\ndecay=0.0005\nmax_batches=500200\nsteps=400000,450000\n"
            source.write_text(head + "\n".join("[convolutional]\nfilters=255\n[yolo]\nclasses=80\n" for _ in range(3)), encoding="utf-8")
            patch_yolov3_cfg(source, output)
            text = output.read_text(encoding="utf-8")
            self.assertIn("max_batches=6000", text); self.assertIn("steps=4800,5400", text)
            self.assertEqual(text.count("classes=1"), 3); self.assertEqual(text.count("filters=18"), 3)

    def test_yolo_thresholds_are_selected_from_validation_records(self):
        records = [{"image_id": "a", "image_size": [100, 100], "ground_truth_bbox_xyxy": [20, 20, 80, 80], "detections": [{"bbox_xyxy_original": [20, 20, 80, 80], "confidence": 0.8}]}]
        report = select_validation_configuration(records, [0.5, 0.9], [0.4], [0.0, 0.2])
        self.assertEqual(report["selection_split"], "validation")
        self.assertEqual(report["selected"]["confidence_threshold"], 0.5)
        self.assertEqual(report["selected"]["margin_fraction"], 0.0)

    def test_cross_dataset_overlap_is_detected(self):
        manifests = [{"dataset_id": "train", "split": "train", "items": [{"image_id": "a", "patient_id": "p1"}]}, {"dataset_id": "external", "split": "external", "items": [{"image_id": "b", "patient_id": "p1"}]}]
        self.assertFalse(audit_manifest_overlap(manifests)["passed"])


class AnnotationTests(unittest.TestCase):
    def test_two_readers_and_adjudication_are_versioned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            initialize_project(root, {"dataset_id": "fitzpatrick_external", "items": [{"image_id": "a", "image_path": "a.jpg", "fitzpatrick": "VI"}]})
            png_path = Path(directory) / "mask.png"; Image.fromarray(np.pad(np.ones((2, 2), np.uint8) * 255, 1)).save(png_path); png = png_path.read_bytes()
            save_mask_version(root, "a", "annotator_1", "lesion", png, actor="r1")
            save_mask_version(root, "a", "annotator_2", "lesion", png, actor="r2")
            self.assertEqual(project_status(root)["counts"]["double_annotated"], 1)
            record = save_mask_version(root, "a", "adjudicator", "lesion", png, actor="adjudicator")
            self.assertEqual(record["version"], 1); self.assertEqual(project_status(root)["counts"]["complete"], 1)
            exported = export_consensus_manifest(root, Path(directory) / "export.json"); self.assertTrue(exported["complete"])


class ReportingTests(unittest.TestCase):
    def test_aggregation_pairing_and_files(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory); atomic_write_json(run / "run_manifest.json", {"run_id": "x"})
            for method, values in (("S01", (0.8, 0.7)), ("S16", (0.6, 0.5))):
                for index, value in enumerate(values):
                    atomic_write_json(run / "predictions" / method / str(index) / "result.json", {"image_id": str(index), "method_id": method, "evaluation": "B1", "failure_code": None, "backend": {"backend_time_ms": 1}, "metrics": {"threshold_jaccard": value, "jaccard": value, "dice": value, "boundary_f1": value}})
            report = write_report(run, repetitions=100, seed=7)
            self.assertEqual(len(report["summaries"]), 2); self.assertTrue(report["pairwise_comparisons"])
            for name in ("metrics_summary.csv", "statistical_comparisons.csv", "metrics_summary.md", "report.json"): self.assertTrue((run / name).is_file())


class SealedIntegrityTests(unittest.TestCase):
    def test_audit_detects_configuration_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config = root / "config.json"; manifest = root / "manifest.json"; checkpoint = root / "weight.pt"
            atomic_write_json(config, {"schema_version": 1}); atomic_write_json(manifest, {"schema_version": 1}); checkpoint.write_bytes(b"weight")
            commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()
            prepared = root / "prepared.json"
            atomic_write_json(prepared, {"status": "prepared", "git_commit": commit, "configuration_path": str(config), "configuration_hash": content_hash({"schema_version": 1}), "dataset_manifest_path": str(manifest), "dataset_manifest_hash": content_hash({"schema_version": 1}), "checkpoints": {"S01": {"members": [{"resolved_path": str(checkpoint), "sha256": sha256_file(checkpoint)}]}}})
            self.assertTrue(sealed_test.audit_prepared(prepared)["passed"])
            atomic_write_json(config, {"schema_version": 1, "threshold": 0.6})
            self.assertIn("configuración modificada", sealed_test.audit_prepared(prepared)["problems"])


if __name__ == "__main__":
    unittest.main()
