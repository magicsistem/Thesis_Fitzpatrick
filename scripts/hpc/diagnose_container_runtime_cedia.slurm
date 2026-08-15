#!/usr/bin/env bash
# CPU-only proof of the runtime actually visible on a Slurm compute node.
#SBATCH --job-name=thesis-container-runtime
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail
: "${PROJECT_ROOT:?}"; : "${SIF_PATH:?}"
PROJECT_ROOT=$(realpath "$PROJECT_ROOT"); SIF_PATH=$(realpath "$SIF_PATH")
cd "$PROJECT_ROOT"
echo "host=$(hostname) job=${SLURM_JOB_ID:-manual} utc=$(date -u +%FT%TZ)"
command -v singularity || true; singularity --version 2>&1 || true
command -v apptainer || true; apptainer --version 2>&1 || true
module avail singularity apptainer 2>&1 || true
exec scripts/hpc/run_in_container.sh --cpu -- python -c '
import json, sys
payload = {"sys_executable": sys.executable, "python": sys.version}
try:
 import torch
 payload.update(torch=torch.__version__, cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available())
except Exception as exc: payload["torch_error"] = repr(exc)
print(json.dumps(payload, sort_keys=True))
'
