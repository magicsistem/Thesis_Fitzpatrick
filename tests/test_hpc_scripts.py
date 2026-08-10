from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
import json


REPO_ROOT = Path(__file__).resolve().parents[1]
HPC_ROOT = REPO_ROOT / "scripts" / "hpc"


class HPCScriptTests(unittest.TestCase):
    def test_scripts_have_valid_bash_syntax(self):
        scripts = sorted(path for path in HPC_ROOT.iterdir() if path.suffix in {".sh", ".slurm"})
        completed = subprocess.run(["bash", "-n", *map(str, scripts)], check=False, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_diagnostic_is_safe_when_runtime_and_sif_are_absent(self):
        with tempfile.TemporaryDirectory() as home:
            env = {**os.environ, "HOME": home, "PATH": "/usr/bin:/bin"}
            env.pop("SIF_PATH", None); env.pop("SIF_SEARCH_ROOTS", None)
            completed = subprocess.run(["bash", str(HPC_ROOT / "inspect_cedia_environment.sh")], env=env, check=False, capture_output=True, text=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("SIF not found", completed.stdout)
        self.assertNotIn("sbatch ", completed.stdout)

    def test_arrays_are_single_gpu_and_mapping_is_explicit(self):
        yolo = (HPC_ROOT / "train_yolo_cedia.slurm").read_text(encoding="utf-8")
        b2 = (HPC_ROOT / "train_b2_cedia.slurm").read_text(encoding="utf-8")
        benchmark = (HPC_ROOT / "run_benchmark_cedia.slurm").read_text(encoding="utf-8")
        gpu_templates = [path.read_text(encoding="utf-8") for path in HPC_ROOT.glob("*.slurm")]
        for text in gpu_templates:
            self.assertIn("#SBATCH --partition=gpu", text)
            self.assertIn("#SBATCH --gres=gpu:a100-sxm4-40gb:1", text)
            self.assertNotIn("conda", text.lower())
            self.assertIn("SIF_PATH", text)
            self.assertIn("run_in_container.sh", text)
        self.assertIn("#SBATCH --array=0-4%2", yolo)
        self.assertIn("FOLD=${SLURM_ARRAY_TASK_ID", yolo)
        self.assertIn("#SBATCH --array=0-74%2", b2)
        self.assertIn("METHOD_INDEX=$((TASK / 5))", b2)
        self.assertIn("FOLD=$((TASK % 5))", b2)
        self.assertIn("validated train-only folds and five frozen YOLO detectors", b2)
        self.assertIn("oof_manifest.json", b2)
        self.assertIn("#SBATCH --array=0-5%1", benchmark)
        self.assertIn("stages=(A B1 C0 C1 C2 C3)", benchmark)

    def test_cedia_files_do_not_assume_laptop_paths_or_remote_conda(self):
        paths = [
            HPC_ROOT / "inspect_cedia_environment.sh",
            HPC_ROOT / "train_yolo_cedia.slurm",
            HPC_ROOT / "train_b2_cedia.slurm",
            REPO_ROOT / "docs" / "CEDIA_OPEN_ONDEMAND.md",
        ]
        text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        self.assertNotIn("/run/media/miguel", text)
        self.assertNotIn("/home/miguel/Downloads", text)
        self.assertNotIn("module load", text)
        self.assertNotRegex(text, re.compile(r"(?:^|\n)\s*(?:ssh|scp|rsync|sftp)\s"))
        for slurm in paths[1:3]:
            content = slurm.read_text(encoding="utf-8").lower()
            self.assertNotIn("conda", content)
            self.assertIn("${sif_path", content)

    def test_bootstrap_dry_run_needs_no_local_sif_and_changes_nothing(self):
        completed = subprocess.run(
            ["bash", str(HPC_ROOT / "bootstrap_cedia.sh"), "--dry-run"],
            cwd=REPO_ROOT, check=False, capture_output=True, text=True,
            env={**os.environ, "SIF_PATH": "/missing/pytorch_24.01-py3.sif"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("DRY RUN", completed.stdout)

    def test_resource_manifest_covers_catalog_and_has_fixed_identities(self):
        resources = json.loads((REPO_ROOT / "configs/hpc/external_resources.json").read_text(encoding="utf-8"))
        catalog = json.loads((REPO_ROOT / "configs/segmentation_models.json").read_text(encoding="utf-8"))["models"]
        checkpoint_ids = {item["id"] for item in resources["checkpoints"]}
        self.assertEqual(len(catalog), 15)
        self.assertTrue({item["id"] for item in catalog}.issubset(checkpoint_ids))
        for item in resources["checkpoints"]:
            self.assertRegex(item["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(item["expected_bytes"], 0)
        for item in resources["sources"]:
            self.assertRegex(item["revision"], r"^[0-9a-f]{40}$")

    def test_catalog_can_switch_from_conda_to_persistent_container_python(self):
        catalog = json.loads((REPO_ROOT / "configs/segmentation_models.json").read_text(encoding="utf-8"))["models"]
        for item in catalog:
            self.assertIn("{device}", item["adapter_command"])
        server = (REPO_ROOT / "scripts/serve_segmentation_review.py").read_text(encoding="utf-8")
        self.assertIn("THESIS_ADAPTER_PYTHON", server)
        self.assertIn("THESIS_DEVICE", server)

    def test_readme_references_existing_hpc_entrypoints(self):
        documentation = "\n".join(
            path.read_text(encoding="utf-8") for path in (
                REPO_ROOT / "README.md", REPO_ROOT / "docs/CEDIA_FROM_ZERO.md",
                REPO_ROOT / "docs/CEDIA_OPEN_ONDEMAND.md",
            )
        )
        names = set(re.findall(r"scripts/hpc/[A-Za-z0-9_.-]+", documentation))
        missing = sorted(name for name in names if not (REPO_ROOT / name).is_file())
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
