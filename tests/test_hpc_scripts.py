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
        gpu_templates = [path.read_text(encoding="utf-8") for path in HPC_ROOT.glob("*.slurm") if "#SBATCH --gres=gpu" in path.read_text(encoding="utf-8")]
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
        validator = (HPC_ROOT / "validate_existing_cedia.slurm").read_text(encoding="utf-8")
        self.assertNotIn("#SBATCH --partition=gpu", validator)
        self.assertNotIn("--gres=gpu", validator)
        self.assertNotIn("bootstrap_cedia", validator)
        self.assertNotIn("preflight_pipeline", validator)

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

    def test_p0_slurm_uses_24_processes_reuse_cache_and_one_thread_libraries(self):
        p0 = (HPC_ROOT / "prepare_p0_oof_cedia.slurm").read_text(encoding="utf-8")
        wrapper = (HPC_ROOT / "run_in_container.sh").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --cpus-per-task=32", p0)
        self.assertIn("#SBATCH --mem=60G", p0)
        self.assertIn("#SBATCH --gres=gpu:a100-sxm4-40gb:1", p0)
        self.assertIn("P0_WORKERS=${P0_WORKERS:-24}", p0)
        self.assertIn('P0_LIMIT=${P0_LIMIT:-}', p0)
        self.assertIn('--workers "$7" --reuse-cache', p0)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
            self.assertIn(name, p0)
            self.assertIn(name, wrapper)

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

    def test_host_entrypoints_never_execute_project_python_directly(self):
        # These files are executed by the login node or Slurm's shell.  Python
        # syntax belongs behind run_in_container.sh; *_inside_container.sh is
        # deliberately excluded because it is the command passed to that wrapper.
        host_entrypoints = (
            "bootstrap_cedia.sh", "validate_existing_cedia.sh",
            "preflight_pipeline_cedia.slurm", "train_yolo_cedia.slurm",
            "train_b2_cedia.slurm", "run_benchmark_cedia.slurm",
            "launch_all_cedia.sh", "launch_pipeline_cedia.sh",
            "validate_existing_cedia.slurm",
        )
        for name in host_entrypoints:
            text = (HPC_ROOT / name).read_text(encoding="utf-8")
            self.assertNotRegex(text, re.compile(r"(?m)^\\s*python(?:3)?\\s"), name)

    def test_validate_existing_uses_container_python_not_an_old_host_python(self):
        source = HPC_ROOT / "validate_existing_cedia.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            hpc = root / "scripts" / "hpc"; hpc.mkdir(parents=True)
            target = hpc / source.name
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            target.chmod(0o755)
            calls, host_python = Path(directory) / "container.calls", Path(directory) / "host-python.calls"
            (hpc / "run_in_container.sh").write_text(
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"$CALL_LOG\"\nexit 0\n",
                encoding="utf-8",
            )
            (hpc / "run_in_container.sh").chmod(0o755)
            fake_bin = Path(directory) / "bin"; fake_bin.mkdir()
            (fake_bin / "python3").write_text(
                "#!/usr/bin/env bash\nprintf invoked >> \"$HOST_PYTHON_LOG\"\nexit 99\n",
                encoding="utf-8",
            )
            (fake_bin / "python3").chmod(0o755)
            data = root / "data"; data.mkdir()
            checkpoint = root / "checkpoint"; checkpoint.write_bytes(b"unchanged-checkpoint")
            dataset = data / "dataset-sentinel"; dataset.write_bytes(b"unchanged-dataset")
            before = (checkpoint.read_bytes(), dataset.read_bytes())
            completed = subprocess.run(
                ["bash", str(target), "1"], cwd=root, check=False, capture_output=True, text=True,
                env={**os.environ, "PROJECT_ROOT": str(root), "DATA_ROOT": str(data),
                     "CALL_LOG": str(calls), "HOST_PYTHON_LOG": str(host_python),
                     "PATH": f"{fake_bin}:{os.environ['PATH']}"},
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse(host_python.exists(), "validate_existing invoked host python3")
            self.assertEqual((checkpoint.read_bytes(), dataset.read_bytes()), before)
            wrapper_calls = calls.read_text(encoding="utf-8")
            self.assertIn("--check-runtime-paths", wrapper_calls)
            self.assertIn("scripts/benchmark/yolov3.py resume-plan", wrapper_calls)
            self.assertIn("--fold 1", wrapper_calls)
            self.assertNotIn("bootstrap_resources.py", wrapper_calls)
            self.assertNotIn("bootstrap_cedia.sh", wrapper_calls)
            self.assertNotIn("prepare_isic2018", wrapper_calls)

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

    def test_monitor_cleanup_never_signals_darknet_and_refuses_ambiguous_identity(self):
        helper = HPC_ROOT / "runtime_diagnostics.sh"
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory); log = project / "diagnostics.log"; calls = project / "kill.calls"
            isolated = (
                f'PROJECT_ROOT="{project}"; source "{helper}"; setsid sleep 30 & apptainer=$!; setsid sleep 30 & darknet=$!; '
                f'runtime_diag_start "{log}" 120 "$apptainer"; monitor=$RUNTIME_DIAGNOSTICS_PID; '
                'test "$monitor" != "$darknet"; test "$monitor" != "$apptainer"; test "$monitor" != "$$"; '
                'test "$RUNTIME_DIAGNOSTICS_PGID" = "$monitor"; test "$RUNTIME_DIAGNOSTICS_SID" = "$monitor"; '
                'runtime_diag_stop; kill -0 "$darknet"; kill -0 "$apptainer"; '
                'kill -TERM "$darknet" "$apptainer"; wait "$darknet" "$apptainer" 2>/dev/null || true'
            )
            completed = subprocess.run(["bash", "-c", isolated], check=False, capture_output=True, text=True, timeout=10)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            ambiguous = (
                f'PROJECT_ROOT="{project}"; RUNTIME_DIAGNOSTICS_LOG="{log}"; source "{helper}"; '
                f'kill() {{ printf "%s\\n" "$*" >> "{calls}"; return 0; }}; '
                'RUNTIME_DIAGNOSTICS_PID=99999; RUNTIME_DIAGNOSTICS_PGID=1; RUNTIME_DIAGNOSTICS_SID=1; RUNTIME_DIAGNOSTICS_MONITOR_STATE=""; '
                'runtime_diag_stop; test ! -e "' + str(calls) + '"'
            )
            refused = subprocess.run(["bash", "-c", ambiguous], check=False, capture_output=True, text=True, timeout=10)
            self.assertEqual(refused.returncode, 0, refused.stderr)

    def test_full_yolo_task_opens_one_container_for_two_attempts(self):
        source = HPC_ROOT / "train_yolo_cedia.slurm"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"; hpc = root / "scripts/hpc"; hpc.mkdir(parents=True)
            task = hpc / source.name; task.write_text(source.read_text(encoding="utf-8"), encoding="utf-8"); task.chmod(0o755)
            (hpc / "runtime_diagnostics.sh").write_text((HPC_ROOT / "runtime_diagnostics.sh").read_text(encoding="utf-8"), encoding="utf-8")
            calls = Path(directory) / "container.calls"; runner = hpc / "run_in_container.sh"
            runner.write_text(f'#!/bin/sh\ncase "$*" in *--check-runtime-paths*) echo check >> "{calls}";; *) printf "exec %s\\n" "$*" >> "{calls}";; esac\n', encoding="utf-8"); runner.chmod(0o755)
            data = root / "data/manifests"; data.mkdir(parents=True)
            for name in ("isic2018_task1_train_disjoint.json", "isic2018_task1_train_disjoint_folds_5.json"):
                (data / name).write_text("{}", encoding="utf-8")
            sif = root / "image.sif"; sif.write_bytes(b"sif")
            darknet = root / "models/yolov3-darknet/source/darknet"; darknet.parent.mkdir(parents=True); darknet.write_bytes(b"x")
            initial = root / "models/yolov3-darknet/checkpoints/darknet53.conv.74"; initial.parent.mkdir(parents=True); initial.write_bytes(b"x")
            fold = root / "results/benchmark_v1/yolo/fold-0"; fold.mkdir(parents=True)
            for name in ("lesion.data", "lesion-yolov3.cfg", "label_audit.json"):
                (fold / name).write_text("x", encoding="utf-8")
            fake = Path(directory) / "bin"; fake.mkdir(); (fake / "git").write_text("#!/bin/sh\necho deadbeef\n", encoding="utf-8"); (fake / "nvidia-smi").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            for path in fake.iterdir(): path.chmod(0o755)
            env = {**os.environ, "PROJECT_ROOT": str(root), "DATA_ROOT": str(root / "data"), "SIF_PATH": str(sif), "DARKNET_GPU_CONFIRMED": "YES", "SLURM_ARRAY_TASK_ID": "0", "SLURM_JOB_ID": "901", "YOLO_MAX_ATTEMPTS": "2", "PATH": f"{fake}:{os.environ['PATH']}"}
            completed = subprocess.run(["bash", str(task)], env=env, cwd=root, check=False, capture_output=True, text=True, timeout=10)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            entries = calls.read_text(encoding="utf-8").splitlines()
            self.assertEqual(entries.count("check"), 1)
            self.assertEqual(sum(entry.startswith("exec ") for entry in entries), 1)
            self.assertIn("--max-attempts 2", entries[-1])

    def test_container_runtime_paths_are_job_local_or_safe_fallback(self):
        runner = HPC_ROOT / "run_in_container.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"; root.mkdir(); slurm_tmp = Path(directory) / "slurm"; slurm_tmp.mkdir()
            env = {**os.environ, "PROJECT_ROOT": str(root), "SLURM_TMPDIR": str(slurm_tmp), "SLURM_JOB_ID": "91", "SLURM_ARRAY_TASK_ID": "2"}
            completed = subprocess.run(["bash", str(runner), "--check-runtime-paths"], env=env, check=False, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            first_tmp = re.search(r"tmp=([^ ]+)", completed.stdout).group(1)
            self.assertIn(f"source=slurm tmp={slurm_tmp}/apptainer-91-2-0/invocation.", completed.stdout)
            self.assertFalse(Path(first_tmp).exists(), "check invocation must clean its own temporary directory")
            repeated = subprocess.run(["bash", str(runner), "--check-runtime-paths"], env=env, check=False, capture_output=True, text=True)
            self.assertNotEqual(first_tmp, re.search(r"tmp=([^ ]+)", repeated.stdout).group(1))
            fallback_env = {**env, "SLURM_TMPDIR": "", "SLURM_JOB_ID": "92"}
            fallback = subprocess.run(["bash", str(runner), "--check-runtime-paths"], env=fallback_env, check=False, capture_output=True, text=True)
            expected = root / ".cedia/apptainer-tmp" / os.environ.get("USER", str(os.getuid())) / "jobs/92-2-0"
            self.assertEqual(fallback.returncode, 0, fallback.stderr)
            self.assertIn(f"source=fallback tmp={expected}/invocation.", fallback.stdout)

    def test_container_invocations_are_concurrent_isolated_cleaned_and_preserve_exit_code(self):
        runner = HPC_ROOT / "run_in_container.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"; root.mkdir(); (root / "data").mkdir(); sif = root / "image.sif"; sif.write_bytes(b"sif")
            venv = root / "venv/bin"; venv.mkdir(parents=True); (venv / "python").write_text("#!/bin/sh\n", encoding="utf-8"); (venv / "python").chmod(0o755)
            fake = Path(directory) / "bin"; fake.mkdir(); calls = Path(directory) / "runtime.calls"
            (fake / "git").write_text("#!/bin/sh\necho deadbeef\n", encoding="utf-8")
            (fake / "apptainer").write_text("#!/bin/sh\n[ \"$1\" = --version ] && { echo apptainer-test; exit 0; }\nprintf '%s\\n' \"$APPTAINER_TMPDIR\" >> \"$CALL_LOG\"\nsleep 0.1\nexit \"${FAKE_EXIT:-0}\"\n", encoding="utf-8")
            for path in fake.iterdir(): path.chmod(0o755)
            env = {**os.environ, "PROJECT_ROOT": str(root), "DATA_ROOT": str(root / "data"), "SIF_PATH": str(sif), "VENV_PATH": str(root / "venv"), "SLURM_TMPDIR": "", "SLURM_JOB_ID": "99", "SLURM_ARRAY_TASK_ID": "3", "CALL_LOG": str(calls), "FAKE_EXIT": "17", "PATH": f"{fake}:{os.environ['PATH']}"}
            failed = subprocess.run(["bash", str(runner), "--cpu", "--", "true"], env=env, check=False, capture_output=True, text=True)
            self.assertEqual(failed.returncode, 17)
            first = calls.read_text(encoding="utf-8").splitlines()[0]
            self.assertFalse(Path(first).exists())
            env["FAKE_EXIT"] = "0"
            jobs = [subprocess.Popen(["bash", str(runner), "--cpu", "--", "true"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
            results = [job.communicate(timeout=10) for job in jobs]
            self.assertTrue(all(job.returncode == 0 for job in jobs), results)
            paths = calls.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(paths), len(set(paths)))
            self.assertTrue(all(not Path(path).exists() for path in paths))
            diagnostic = root / "diagnostics.log"
            diagnosed = subprocess.run(["bash", str(runner), "--cpu", "--", "true"], env={**env, "RUNTIME_DIAGNOSTICS_LOG": str(diagnostic)}, check=False, capture_output=True, text=True)
            self.assertEqual(diagnosed.returncode, 0, diagnosed.stderr)
            diagnostic_text = diagnostic.read_text(encoding="utf-8")
            self.assertIn("container_runtime", diagnostic_text)
            self.assertIn("label=start", diagnostic_text)
            self.assertRegex(diagnostic_text, r"target_pid=\d+")

    def test_runtime_prefers_singularity_skips_sif_hash_and_nested_wrapper_is_direct(self):
        runner = HPC_ROOT / "run_in_container.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"; root.mkdir(); (root / "data").mkdir(); sif = root / "image.sif"; sif.write_bytes(b"sif")
            fake = Path(directory) / "bin"; fake.mkdir(); calls = Path(directory) / "runtime.calls"
            (fake / "git").write_text("#!/bin/sh\necho deadbeef\n", encoding="utf-8")
            (fake / "sha256sum").write_text("#!/bin/sh\necho hash-must-not-run >&2; exit 99\n", encoding="utf-8")
            (fake / "singularity").write_text(f"#!/bin/sh\n[ \"$1\" = --version ] && {{ echo singularity-3; exit 0; }}; echo singularity >> {calls}; exit 0\n", encoding="utf-8")
            (fake / "apptainer").write_text(f"#!/bin/sh\necho apptainer >> {calls}; exit 0\n", encoding="utf-8")
            for path in fake.iterdir(): path.chmod(0o755)
            env = {**os.environ, "PROJECT_ROOT": str(root), "DATA_ROOT": str(root / "data"), "SIF_PATH": str(sif), "PATH": f"{fake}:{os.environ['PATH']}"}
            completed = subprocess.run(["bash", str(runner), "--cpu", "--no-venv", "--", "true"], env=env, check=False, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(calls.read_text(encoding="utf-8").strip(), "singularity")
            self.assertIn("kind=singularity", completed.stdout)
            nested = subprocess.run(["bash", str(runner), "--", "false"], env={**env, "THESIS_IN_CONTAINER": "1"}, check=False, capture_output=True, text=True)
            self.assertEqual(nested.returncode, 1)
            self.assertEqual(calls.read_text(encoding="utf-8").strip(), "singularity")

            fallback = Path(directory) / "fallback"; fallback.mkdir()
            (fallback / "git").write_text("#!/bin/sh\necho deadbeef\n", encoding="utf-8")
            (fallback / "apptainer").write_text(f"#!/bin/sh\n[ \"$1\" = --version ] && {{ echo apptainer-1; exit 0; }}; echo apptainer >> {calls}; exit 0\n", encoding="utf-8")
            for path in fallback.iterdir(): path.chmod(0o755)
            fallback_run = subprocess.run(["bash", str(runner), "--cpu", "--no-venv", "--", "true"], env={**env, "PATH": f"{fallback}:{os.environ['PATH']}"}, check=False, capture_output=True, text=True)
            self.assertEqual(fallback_run.returncode, 0, fallback_run.stderr)
            self.assertIn("kind=apptainer-fallback", fallback_run.stdout)
            absent = subprocess.run(["bash", str(runner), "--cpu", "--no-venv", "--", "true"], env={**env, "PATH": "/usr/bin:/bin"}, check=False, capture_output=True, text=True)
            self.assertEqual(absent.returncode, 2)
            self.assertIn("Neither singularity nor apptainer", absent.stderr)

    def test_runtime_diagnostic_is_cpu_only_and_validate_script_never_bootstraps(self):
        diagnostic = (HPC_ROOT / "diagnose_container_runtime_cedia.slurm").read_text(encoding="utf-8")
        self.assertNotIn("--partition=gpu", diagnostic)
        self.assertNotIn("--gres=", diagnostic)
        self.assertIn("run_in_container.sh --cpu -- python", diagnostic)
        self.assertIn("module avail singularity apptainer", diagnostic)
        validate = (HPC_ROOT / "validate_existing_cedia.sh").read_text(encoding="utf-8")
        self.assertNotIn("bootstrap_resources.py", validate)
        self.assertEqual(validate.count("run_in_container.sh --cpu -- python"), 1)

    def test_resume_launcher_submits_only_afterok_without_preflight_and_rejects_duplicate(self):
        source = HPC_ROOT / "launch_pipeline_cedia.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"; hpc = root / "scripts/hpc"; hpc.mkdir(parents=True)
            target = hpc / source.name; target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8"); target.chmod(0o755)
            data = root / "data/manifests"; data.mkdir(parents=True)
            for name in ("isic2018_task1_train_disjoint.json", "isic2018_task1_validation_disjoint.json", "isic2018_task1_train_disjoint_folds_5.json"):
                (data / name).write_text("{}", encoding="utf-8")
            (root / "image.sif").write_bytes(b"sif"); (root / "results/benchmark_v1/yolo").mkdir(parents=True)
            fake = Path(directory) / "bin"; fake.mkdir(); calls = Path(directory) / "sbatch.calls"
            (fake / "git").write_text("#!/bin/sh\ncase \"$*\" in *'--abbrev-ref HEAD'*) echo fix/hpc-pipeline-validation;; *'status --porcelain'*) :;; *'rev-parse HEAD'*) echo deadbeef;; *'merge-base'*) exit 0;; esac\n", encoding="utf-8")
            (fake / "squeue").write_text("#!/bin/sh\n[ -n \"${SQUEUE_ACTIVE:-}\" ] && echo \"$SQUEUE_ACTIVE\"\n", encoding="utf-8")
            (fake / "sbatch").write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {calls}\nprintf '23%s;cluster\\n' \"$(wc -l < {calls})\"\n", encoding="utf-8")
            for path in fake.iterdir(): path.chmod(0o755)
            env = {**os.environ, "PROJECT_ROOT": str(root), "DATA_ROOT": str(root / "data"), "SIF_PATH": str(root / "image.sif"), "PATH": f"{fake}:{os.environ['PATH']}", "HOME": directory}
            started = time.monotonic(); completed = subprocess.run(["bash", str(target), "--resume-existing", "--skip-preflight"], env=env, cwd=root, check=False, capture_output=True, text=True, timeout=10)
            self.assertEqual(completed.returncode, 0, completed.stderr); self.assertLess(time.monotonic() - started, 2)
            submitted = calls.read_text(encoding="utf-8").splitlines(); self.assertEqual(len(submitted), 8)
            self.assertNotIn("preflight_pipeline", "\n".join(submitted)); self.assertIn("--array=0,2,3,4%2", submitted[1]); self.assertIn("--array=0-74%2", submitted[4])
            manifest = next((root / ".cedia").glob("resume_pipeline_*.json")); payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertTrue(payload["skip_preflight"]); self.assertTrue(payload["fold_1_reused"]); self.assertEqual(payload["yolo_folds"], "0,2,3,4"); self.assertEqual(len(payload["jobs"]), 8)
            self.assertTrue(all(item["dependency"] is None or item["dependency"].startswith("afterok:") for item in payload["jobs"]))
            self.assertEqual([item["dependency"] for item in payload["jobs"]], [None, "afterok:231", "afterok:232", "afterok:233", "afterok:234", "afterok:235", "afterok:236", "afterok:237"])
            duplicate = subprocess.run(["bash", str(target), "--resume-existing", "--skip-preflight"], env={**env, "SQUEUE_ACTIVE": "2301"}, cwd=root, check=False, capture_output=True, text=True)
            self.assertEqual(duplicate.returncode, 2); self.assertIn("active", duplicate.stderr.lower())
            (root / "results/benchmark_v1/preprocessing").mkdir(parents=True); (root / "results/benchmark_v1/preprocessing/oof_manifest.json").write_text("{}", encoding="utf-8")
            completed_result = subprocess.run(["bash", str(target), "--resume-existing", "--skip-preflight"], env=env, cwd=root, check=False, capture_output=True, text=True)
            self.assertEqual(completed_result.returncode, 2); self.assertIn("Existing downstream result", completed_result.stderr)

    def test_recovery_launcher_requires_explicit_pending_folds_and_optional_node_exclusion(self):
        launcher = (HPC_ROOT / "launch_pipeline_cedia.sh").read_text(encoding="utf-8")
        self.assertIn("--recover-existing", launcher)
        self.assertIn("YOLO_EXPECTED_PENDING", launcher)
        self.assertIn("--exclude=\"$YOLO_EXCLUDE_NODES\"", launcher)
        self.assertIn('YOLO_ARRAY="$PENDING_FOLDS%2"', launcher)
        self.assertNotIn("preflight_pipeline", launcher)
        self.assertIn('folds 1 and 2 are completed/reused', launcher)

    def test_recovery_submits_one_validation_and_refuses_old_active_or_blocked_jobs(self):
        source = HPC_ROOT / "launch_pipeline_cedia.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"; hpc = root / "scripts/hpc"; hpc.mkdir(parents=True)
            target = hpc / source.name; target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8"); target.chmod(0o755)
            data = root / "data/manifests"; data.mkdir(parents=True)
            for name in ("isic2018_task1_train_disjoint.json", "isic2018_task1_validation_disjoint.json", "isic2018_task1_train_disjoint_folds_5.json"):
                (data / name).write_text("{}", encoding="utf-8")
            (root / "image.sif").write_bytes(b"sif"); (root / "results/benchmark_v1/yolo").mkdir(parents=True)
            fake = Path(directory) / "bin"; fake.mkdir(); calls = Path(directory) / "sbatch.calls"; cancelled = Path(directory) / "scancel.calls"
            (fake / "git").write_text("#!/bin/sh\ncase \"$*\" in *'--abbrev-ref HEAD'*) echo fix/hpc-pipeline-validation;; *'status --porcelain'*) :;; *'rev-parse HEAD'*) echo deadbeef;; *'merge-base'*) exit 0;; esac\n", encoding="utf-8")
            (fake / "squeue").write_text("#!/bin/sh\ncase \"$*\" in *'23063'*) [ -z \"${PREVIOUS_ACTIVE:-}\" ] || echo '23063_0 RUNNING node';; *'23064,23065,23066,23067,23068,23069'*) [ -z \"${BLOCKED_ACTIVE:-}\" ] || echo '23064 PENDING Dependency';; esac\n", encoding="utf-8")
            (fake / "sbatch").write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {calls}\nprintf '31%s;cluster\\n' \"$(wc -l < {calls})\"\n", encoding="utf-8")
            (fake / "scancel").write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {cancelled}\n", encoding="utf-8")
            for path in fake.iterdir(): path.chmod(0o755)
            env = {**os.environ, "PROJECT_ROOT": str(root), "DATA_ROOT": str(root / "data"), "SIF_PATH": str(root / "image.sif"), "PATH": f"{fake}:{os.environ['PATH']}", "HOME": directory}
            args = ["bash", str(target), "--recover-existing", "--skip-preflight", "--folds", "0,3", "--previous-yolo-job", "23063", "--blocked-jobs", "23064,23065,23066,23067,23068,23069"]
            previous = subprocess.run(args, env={**env, "PREVIOUS_ACTIVE": "1"}, cwd=root, check=False, capture_output=True, text=True)
            self.assertEqual(previous.returncode, 2); self.assertIn("23063", previous.stderr); self.assertFalse(calls.exists())
            blocked = subprocess.run(args, env={**env, "BLOCKED_ACTIVE": "1"}, cwd=root, check=False, capture_output=True, text=True)
            self.assertEqual(blocked.returncode, 2); self.assertIn("23064,23065,23066,23067,23068,23069", blocked.stderr); self.assertIn("scancel 23064 23065 23066 23067 23068 23069", blocked.stderr); self.assertFalse(calls.exists())
            completed = subprocess.run(args, env=env, cwd=root, check=False, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            submitted = calls.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(submitted), 8)
            self.assertEqual(sum("validate_existing_cedia.slurm" in item for item in submitted), 1)
            self.assertIn("--array=0,3%2", submitted[1])
            self.assertFalse(cancelled.exists(), "launcher must never call scancel")

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
        self.assertEqual(yolo.count('run_in_container.sh" -- python scripts/benchmark/yolov3.py train'), 1)
        self.assertIn('--max-attempts "$MAX_ATTEMPTS"', yolo)
        self.assertNotIn('for attempt in $(seq', yolo)

    def test_b2_and_benchmark_keep_preconditions_inside_one_container(self):
        for name in ("train_b2_cedia.slurm", "run_benchmark_cedia.slurm"):
            text = (HPC_ROOT / name).read_text(encoding="utf-8")
            self.assertEqual(text.count("run_in_container.sh"), 1, name)
            self.assertIn("bash -c '", text, name)

    def test_yolo_slurm_delegates_checkpoint_validation_to_python(self):
        training = (HPC_ROOT / "train_yolo_cedia.slurm").read_text(encoding="utf-8")
        self.assertIn('mkdir -p "$FOLD_DIR/backup"', training)
        self.assertIn('--fold "$FOLD"', training)
        self.assertNotIn("find \"$FOLD_DIR/backup\"", training)
        for name in ("prepare_p0_oof_cedia.slurm", "train_b2_cedia.slurm", "run_benchmark_cedia.slurm"):
            self.assertIn("yolov3.py validate-folds", (HPC_ROOT / name).read_text(encoding="utf-8"), name)


if __name__ == "__main__":
    unittest.main()
