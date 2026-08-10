#!/usr/bin/env bash
#SBATCH --job-name=thesis-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail
trap 'rc=$?; echo "smoke failed rc=$rc line=$LINENO utc=$(date -u +%FT%TZ)" >&2; exit "$rc"' ERR
: "${PROJECT_ROOT:?Export PROJECT_ROOT before sbatch}"
: "${DATA_ROOT:?Export DATA_ROOT before sbatch}"
: "${SMOKE_MANIFEST:?Export a non-test development manifest}"
PROJECT_ROOT=$(realpath "$PROJECT_ROOT")
DATA_ROOT=$(realpath "$DATA_ROOT")
SMOKE_MANIFEST=$(realpath "$SMOKE_MANIFEST")
export DATA_ROOT SIF_PATH=${SIF_PATH:-$HOME/pytorch_24.01-py3.sif}

echo "job=$SLURM_JOB_ID host=$(hostname) utc=$(date -u +%FT%TZ) git=$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
cd "$PROJECT_ROOT"
exec scripts/hpc/run_in_container.sh -- \
    python scripts/hpc/smoke_test.py --manifest "$SMOKE_MANIFEST" --data-root "$DATA_ROOT" --device cuda
