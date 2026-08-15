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
P0_WORKERS=${P0_WORKERS:-24}
P0_LIMIT=${P0_LIMIT:-}
[[ "$P0_WORKERS" =~ ^[1-9][0-9]*$ && "$P0_WORKERS" -le "${SLURM_CPUS_PER_TASK:-32}" ]] || { echo "P0_WORKERS must be 1..SLURM_CPUS_PER_TASK" >&2; exit 2; }
[[ -z "$P0_LIMIT" || "$P0_LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo "P0_LIMIT must be empty or positive" >&2; exit 2; }
for variable in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS BLIS_NUM_THREADS; do export "$variable"=1; done
export DATA_ROOT SIF_PATH=${SIF_PATH:-$HOME/pytorch_24.01-py3.sif}
cd "$PROJECT_ROOT"
echo "job=$SLURM_JOB_ID host=$(hostname) utc=$(date -u +%FT%TZ) git=$(git rev-parse HEAD)"
nvidia-smi
exec scripts/hpc/run_in_container.sh -- bash -c '
set -Eeuo pipefail
python scripts/benchmark/yolov3.py validate-folds --root "$1"
extra=()
[[ -z "$6" ]] || extra=(--limit "$6")
exec python scripts/benchmark/prepare_p0_oof.py --manifest "$2" --data-root "$3" --folds "$4" \
    --yolo-root "$1" --artifact-root "$5" --workers "$7" --reuse-cache --confirm-run "${extra[@]}"
' _ "$YOLO_FROZEN_ROOT" "$TRAIN_MANIFEST" "$DATA_ROOT" "$FOLDS_FILE" "$P0_ROOT" "$P0_LIMIT" "$P0_WORKERS"
