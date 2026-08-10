from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
HPC_ROOT = REPO_ROOT / "scripts" / "hpc"


class HPCScriptTests(unittest.TestCase):
    def test_scripts_have_valid_bash_syntax(self):
        scripts = sorted(HPC_ROOT.iterdir())
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
        for text in (yolo, b2):
            self.assertIn("#SBATCH --partition=gpu", text)
            self.assertIn("#SBATCH --gres=gpu:a100-sxm4-40gb:1", text)
            self.assertNotIn("conda", text.lower())
            self.assertIn("SIF_PATH", text)
            self.assertIn("--nv", text)
        self.assertIn("#SBATCH --array=0-4%2", yolo)
        self.assertIn("FOLD=${SLURM_ARRAY_TASK_ID", yolo)
        self.assertIn("#SBATCH --array=0-74%2", b2)
        self.assertIn("METHOD_INDEX=$((TASK / 5))", b2)
        self.assertIn("FOLD=$((TASK % 5))", b2)
        self.assertIn("validated train-only folds and five frozen YOLO detectors", b2)

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
        self.assertNotRegex(text, re.compile(r"\b(?:ssh|scp|rsync|sftp)\b"))
        for slurm in paths[1:3]:
            content = slurm.read_text(encoding="utf-8").lower()
            self.assertNotIn("conda", content)
            self.assertIn("${sif_path", content)


if __name__ == "__main__":
    unittest.main()
