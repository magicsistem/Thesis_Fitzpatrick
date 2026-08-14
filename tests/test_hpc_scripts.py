from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
import json
import time


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

    def test_every_a100_submission_satisfies_cedia_gpu_qos_floor(self):
        templates = list(HPC_ROOT.glob("*.slurm")) + [
            HPC_ROOT / "launch_all_cedia.sh"
        ]
        for path in templates:
            text = path.read_text(encoding="utf-8")
            if "gpu:a100-sxm4-40gb:1" not in text:
                continue
            cpu_requests = [
                int(value)
                for value in re.findall(r"--cpus-per-task=(\d+)", text)
            ]
            memory_requests = [
                int(value)
                for value in re.findall(r"--mem=(\d+)G", text)
            ]
            self.assertTrue(cpu_requests, path)
            self.assertTrue(memory_requests, path)
            self.assertTrue(all(value >= 32 for value in cpu_requests), path)
            self.assertTrue(all(value >= 60 for value in memory_requests), path)

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

    def test_bootstrap_verifies_avit_runtime_after_model_setup(self):
        bootstrap = (HPC_ROOT / "bootstrap_inside_container.sh").read_text(encoding="utf-8")
        setup_index = bootstrap.index('python "scripts/$setup_script"')
        verify_index = bootstrap.index("python scripts/hpc/verify_avit_runtime.py")
        self.assertGreater(verify_index, setup_index)

        requirements = (REPO_ROOT / "configs/hpc/requirements-cuda.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("einops==", requirements)

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

    def test_launcher_dependency_graph_uses_afterok_for_every_edge(self):
        launcher = (HPC_ROOT / "launch_all_cedia.sh").read_text(encoding="utf-8")
        expected = {
            "SMOKE_JOB": 'afterok:$PREFLIGHT_JOB',
            "YOLO_JOB": 'afterok:$SMOKE_JOB',
            "FINALIZE_YOLO_JOB": 'afterok:$YOLO_JOB',
            "P0_JOB": 'afterok:$FINALIZE_YOLO_JOB',
            "B2_JOB": 'afterok:$P0_JOB',
            "CONFIG_JOB": 'afterok:$FINALIZE_YOLO_JOB',
            "BENCHMARK_JOB": 'afterok:$CONFIG_JOB',
            "B2_FINAL_JOB": 'afterok:$B2_JOB:$CONFIG_JOB',
        }
        for variable, dependency in expected.items():
            self.assertRegex(launcher, rf"{variable}=\$\(submit {variable} --dependency=\"{re.escape(dependency)}\"")

    def test_launcher_resolves_explicit_project_from_spooled_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            spool = Path(directory) / "var/spool/slurm"; spool.mkdir(parents=True)
            completed = subprocess.run(
                ["bash", str(HPC_ROOT / "launch_all_cedia.sh")], cwd=spool,
                env={**os.environ, "PROJECT_ROOT": str(REPO_ROOT), "PIPELINE_BRANCH": "deliberately-wrong"},
                check=False, capture_output=True, text=True,
            )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("Expected branch deliberately-wrong", completed.stderr)
        self.assertNotIn("/var/spool/slurm/.cedia", completed.stderr)

    def test_launcher_exits_after_submission_and_does_not_wait(self):
        launcher = (HPC_ROOT / "launch_all_cedia.sh").read_text(encoding="utf-8")
        self.assertTrue(launcher.rstrip().endswith("exit 0"))
        self.assertNotIn("squeue -u", launcher)
        self.assertNotIn("tee", launcher)
        self.assertNotRegex(launcher, r"\b(?:sbatch\s+--wait|wait|sleep|tail\s+-f)\b")

    def test_launcher_submits_quickly_atomically_without_surviving_helpers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"; hpc = root / "scripts/hpc"; hpc.mkdir(parents=True)
            target = hpc / "launch_all_cedia.sh"; target.write_text((HPC_ROOT / "launch_all_cedia.sh").read_text(encoding="utf-8"), encoding="utf-8"); target.chmod(0o755)
            fake = Path(directory) / "bin"; fake.mkdir(); calls = Path(directory) / "sbatch.calls"
            (fake / "git").write_text("#!/bin/sh\ncase \"$*\" in *'--abbrev-ref HEAD'*) echo fix/hpc-pipeline-validation;; *'status --porcelain'*) :;; *'rev-parse HEAD'*) echo deadbeef;; *'merge-base'*) exit 0;; esac\n", encoding="utf-8")
            (fake / "sbatch").write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {calls}\nprintf '23%s;cluster\\n' \"$(wc -l < {calls})\"\n", encoding="utf-8")
            for executable in fake.iterdir(): executable.chmod(0o755)
            started = time.monotonic()
            completed = subprocess.run(["bash", str(target)], cwd=Path(directory), env={**os.environ, "PROJECT_ROOT": str(root), "PATH": f"{fake}:{os.environ['PATH']}", "HOME": directory}, check=False, capture_output=True, text=True, timeout=10)
            elapsed = time.monotonic() - started
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertLess(elapsed, 2.0)
            self.assertEqual(len(calls.read_text(encoding="utf-8").splitlines()), 9)
            registry = next((root / ".cedia").glob("cedia_job_chain_*.txt"))
            self.assertIn("status=submitted", registry.read_text(encoding="utf-8"))
            self.assertFalse(list((root / ".cedia").glob(".cedia_job_chain.*")))

    def test_runtime_monitor_stops_without_retaining_a_child_or_tee(self):
        helper = HPC_ROOT / "runtime_diagnostics.sh"
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory); log = project / "diagnostics.log"
            command = f'PROJECT_ROOT="{project}"; source "{helper}"; runtime_diag_start "{log}" 120; pid=$RUNTIME_DIAGNOSTICS_PID; runtime_diag_stop; kill -0 "$pid" 2>/dev/null && exit 9 || exit 0'
            started = time.monotonic()
            completed = subprocess.run(["bash", "-c", command], check=False, capture_output=True, text=True, timeout=10)
            elapsed = time.monotonic() - started
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertLess(elapsed, 5.0)
            self.assertIn("label=start", log.read_text(encoding="utf-8"))
        self.assertNotIn("\ntee ", helper.read_text(encoding="utf-8"))

    def test_container_runtime_paths_are_job_local_or_safe_fallback(self):
        runner = HPC_ROOT / "run_in_container.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"; root.mkdir(); slurm_tmp = Path(directory) / "slurm"; slurm_tmp.mkdir()
            env = {**os.environ, "PROJECT_ROOT": str(root), "SLURM_TMPDIR": str(slurm_tmp), "SLURM_JOB_ID": "91", "SLURM_ARRAY_TASK_ID": "2"}
            completed = subprocess.run(["bash", str(runner), "--check-runtime-paths"], env=env, check=False, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(f"source=slurm tmp={slurm_tmp}/apptainer-91-2-0", completed.stdout)
            self.assertTrue((slurm_tmp / "apptainer-91-2-0").is_dir())
            fallback_env = {**env, "SLURM_TMPDIR": "", "SLURM_JOB_ID": "92"}
            fallback = subprocess.run(["bash", str(runner), "--check-runtime-paths"], env=fallback_env, check=False, capture_output=True, text=True)
            expected = root / ".cedia/apptainer-tmp" / os.environ.get("USER", str(os.getuid())) / "jobs/92-2-0"
            self.assertEqual(fallback.returncode, 0, fallback.stderr)
            self.assertIn(f"source=fallback tmp={expected}", fallback.stdout)

    def test_container_paths_report_unwritable_and_cache_is_concurrent_without_locks(self):
        runner = HPC_ROOT / "run_in_container.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"; blocked = root / ".cedia"; blocked.mkdir(parents=True); blocked.chmod(0o500)
            env = {**os.environ, "PROJECT_ROOT": str(root), "SLURM_TMPDIR": "", "SLURM_JOB_ID": "93"}
            failed = subprocess.run(["bash", str(runner), "--check-runtime-paths"], env=env, check=False, capture_output=True, text=True)
            blocked.chmod(0o700)
            self.assertEqual(failed.returncode, 2)
            self.assertIn("APPTAINER_TMPDIR", failed.stderr)
            cache_root = root / ".cedia/apptainer-cache" / os.environ.get("USER", str(os.getuid())); cache_root.mkdir(parents=True)
            orphan = cache_root / "orphan.lock"; orphan.write_text("not owned by this pipeline", encoding="utf-8")
            commands = []
            for job in ("94", "95"):
                commands.append(subprocess.Popen(["bash", str(runner), "--check-runtime-paths"], env={**env, "SLURM_JOB_ID": job}, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
            results = [command.communicate(timeout=10) for command in commands]
            self.assertTrue(all(command.returncode == 0 for command in commands), results)
            self.assertTrue(orphan.exists())

    def test_yolo_configuration_failure_precedes_attempt_and_only_signal_75_retries(self):
        yolo = (HPC_ROOT / "train_yolo_cedia.slurm").read_text(encoding="utf-8")
        self.assertLess(yolo.index("--check-runtime-paths"), yolo.index("yolo_attempt="))
        self.assertIn('[[ "$rc" != 75 ]] && exit "$rc"', yolo)

    def test_yolo_slurm_delegates_checkpoint_validation_to_python(self):
        training = (HPC_ROOT / "train_yolo_cedia.slurm").read_text(encoding="utf-8")
        self.assertIn('mkdir -p "$FOLD_DIR/backup"', training)
        self.assertIn('--fold "$FOLD"', training)
        self.assertNotIn("find \"$FOLD_DIR/backup\"", training)
        for name in ("prepare_p0_oof_cedia.slurm", "train_b2_cedia.slurm", "run_benchmark_cedia.slurm"):
            self.assertIn("yolov3.py validate-folds", (HPC_ROOT / name).read_text(encoding="utf-8"), name)


if __name__ == "__main__":
    unittest.main()
