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
export DATA_ROOT SIF_PATH=${SIF_PATH:-$HOME/pytorch_24.01-py3.sif}
cd "$PROJECT_ROOT"
echo "job=$SLURM_JOB_ID host=$(hostname) utc=$(date -u +%FT%TZ) git=$(git rev-parse HEAD)"
nvidia-smi
exec scripts/hpc/run_in_container.sh -- bash -c '
set -Eeuo pipefail
python scripts/benchmark/yolov3.py validate-folds --root "$1"
exec python scripts/benchmark/prepare_p0_oof.py --manifest "$2" --data-root "$3" --folds "$4" \
    --yolo-root "$1" --artifact-root "$5" --confirm-run
' _ "$YOLO_FROZEN_ROOT" "$TRAIN_MANIFEST" "$DATA_ROOT" "$FOLDS_FILE" "$P0_ROOT"
