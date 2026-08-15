#!/usr/bin/env bash
# Read-only CPU gate for the resumed CEDIA DAG.  It never bootstraps or trains.
#SBATCH --job-name=thesis-resume-validate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail
: "${PROJECT_ROOT:?}"; : "${DATA_ROOT:?}"; : "${SIF_PATH:?}"
PROJECT_ROOT=$(realpath "$PROJECT_ROOT"); DATA_ROOT=$(realpath "$DATA_ROOT"); SIF_PATH=$(realpath "$SIF_PATH")
TRAIN_MANIFEST=${TRAIN_MANIFEST:-$DATA_ROOT/manifests/isic2018_task1_train_disjoint.json}
FOLDS_FILE=${FOLDS_FILE:-$DATA_ROOT/manifests/isic2018_task1_train_disjoint_folds_5.json}
YOLO_ROOT=${YOLO_ROOT:-$PROJECT_ROOT/results/benchmark_v1/yolo}
DARKNET_BIN=${DARKNET_BIN:-$PROJECT_ROOT/models/yolov3-darknet/source/darknet}
INITIAL_WEIGHTS=${INITIAL_WEIGHTS:-$PROJECT_ROOT/models/yolov3-darknet/checkpoints/darknet53.conv.74}
for path in "$PROJECT_ROOT" "$DATA_ROOT" "$SIF_PATH" "$TRAIN_MANIFEST" "$FOLDS_FILE" "$YOLO_ROOT" "$DARKNET_BIN" "$INITIAL_WEIGHTS"; do
    [[ -e "$path" ]] || { echo "Missing read-only validation prerequisite: $path" >&2; exit 2; }
done
export PROJECT_ROOT DATA_ROOT SIF_PATH TRAIN_MANIFEST FOLDS_FILE YOLO_ROOT DARKNET_BIN INITIAL_WEIGHTS
cd "$PROJECT_ROOT"
echo "job=${SLURM_JOB_ID:-manual} host=$(hostname) utc=$(date -u +%FT%TZ) gpu=none"
scripts/hpc/run_in_container.sh --check-runtime-paths
exec scripts/hpc/run_in_container.sh --cpu -- python scripts/benchmark/yolov3.py validate-existing \
    --darknet "$DARKNET_BIN" --manifest "$TRAIN_MANIFEST" --folds "$FOLDS_FILE" \
    --initial-weights "$INITIAL_WEIGHTS" --root "$YOLO_ROOT"
