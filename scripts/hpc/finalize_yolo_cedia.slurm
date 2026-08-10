#!/usr/bin/env bash
#SBATCH --job-name=thesis-yolo-finalize
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --time=08:00:00
#SBATCH --array=0-4%2
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -Eeuo pipefail
trap 'rc=$?; echo "YOLO finalization failed rc=$rc line=$LINENO" >&2; exit "$rc"' ERR
: "${PROJECT_ROOT:?}"; : "${DATA_ROOT:?}"; : "${TRAIN_MANIFEST:?}"; : "${FOLDS_FILE:?}"
PROJECT_ROOT=$(realpath "$PROJECT_ROOT"); DATA_ROOT=$(realpath "$DATA_ROOT")
FOLD=${SLURM_ARRAY_TASK_ID:?}; YOLO_ROOT=${YOLO_ROOT:-$PROJECT_ROOT/results/benchmark_v1/yolo}
[[ ! -f "$YOLO_ROOT/fold-$FOLD/frozen.json" ]] || { echo "Fold already frozen; refusing to overwrite" >&2; exit 2; }
export DATA_ROOT SIF_PATH=${SIF_PATH:-$HOME/pytorch_24.01-py3.sif}
cd "$PROJECT_ROOT"
echo "job=$SLURM_JOB_ID fold=$FOLD host=$(hostname) utc=$(date -u +%FT%TZ) git=$(git rev-parse HEAD)"
nvidia-smi
exec scripts/hpc/run_in_container.sh -- python scripts/benchmark/finalize_yolo_fold.py \
    --fold-root "$YOLO_ROOT/fold-$FOLD" --fold "$FOLD" --manifest "$TRAIN_MANIFEST" \
    --data-root "$DATA_ROOT" --folds "$FOLDS_FILE"
