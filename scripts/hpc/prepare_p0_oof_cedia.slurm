#!/usr/bin/env bash
#SBATCH --job-name=thesis-p0-oof
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail
trap 'rc=$?; echo "OOF P0 failed rc=$rc line=$LINENO" >&2; exit "$rc"' ERR
: "${PROJECT_ROOT:?}"; : "${DATA_ROOT:?}"; : "${TRAIN_MANIFEST:?}"; : "${FOLDS_FILE:?}"; : "${YOLO_FROZEN_ROOT:?}"
PROJECT_ROOT=$(realpath "$PROJECT_ROOT"); DATA_ROOT=$(realpath "$DATA_ROOT")
P0_ROOT=${P0_ROOT:-$PROJECT_ROOT/results/benchmark_v1/preprocessing}
[[ ! -f "$P0_ROOT/oof_manifest.json" ]] || { echo "OOF P0 manifest already exists; use --reuse-cache interactively only after auditing it" >&2; exit 2; }
export DATA_ROOT SIF_PATH=${SIF_PATH:-$HOME/pytorch_24.01-py3.sif}
cd "$PROJECT_ROOT"
echo "job=$SLURM_JOB_ID host=$(hostname) utc=$(date -u +%FT%TZ) git=$(git rev-parse HEAD)"
nvidia-smi
exec scripts/hpc/run_in_container.sh -- python scripts/benchmark/prepare_p0_oof.py \
    --manifest "$TRAIN_MANIFEST" --data-root "$DATA_ROOT" --folds "$FOLDS_FILE" \
    --yolo-root "$YOLO_FROZEN_ROOT" --artifact-root "$P0_ROOT" --confirm-run
