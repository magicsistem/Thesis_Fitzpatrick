#!/usr/bin/env bash
#SBATCH --job-name=thesis-yolo-final
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail
trap 'rc=$?; echo "post05 final YOLO train failed rc=$rc line=$LINENO utc=$(date -u +%FT%TZ)" >&2; exit "$rc"' ERR
: "${PROJECT_ROOT:?Export PROJECT_ROOT}"
: "${DATA_ROOT:?Export DATA_ROOT}"
: "${SIF_PATH:?Export SIF_PATH}"
: "${DARKNET_GPU_CONFIRMED:?Export DARKNET_GPU_CONFIRMED=YES after the existing GPU preflight}"
[[ "$DARKNET_GPU_CONFIRMED" == YES ]] || { echo "Darknet GPU build has not been explicitly confirmed" >&2; exit 2; }
PROJECT_ROOT=$(realpath "$PROJECT_ROOT")
DATA_ROOT=$(realpath "$DATA_ROOT")
SIF_PATH=$(realpath "$SIF_PATH")
FINAL_YOLO_ROOT=${FINAL_YOLO_ROOT:-$PROJECT_ROOT/results/benchmark_v1/yolo/final}
DARKNET_BIN=${DARKNET_BIN:-$PROJECT_ROOT/models/yolov3-darknet/source/darknet}
INITIAL_WEIGHTS=${INITIAL_WEIGHTS:-$PROJECT_ROOT/models/yolov3-darknet/checkpoints/darknet53.conv.74}

for path in "$PROJECT_ROOT" "$DATA_ROOT" "$SIF_PATH" "$FINAL_YOLO_ROOT/prepare_manifest.json" "$FINAL_YOLO_ROOT/lesion.data" "$FINAL_YOLO_ROOT/lesion-yolov3-final.cfg" "$DARKNET_BIN" "$INITIAL_WEIGHTS"; do
  [[ -e "$path" ]] || { echo "Missing prerequisite: $path" >&2; exit 2; }
done
[[ ! -e "$FINAL_YOLO_ROOT/frozen.json" ]] || { echo "Final YOLO is already frozen; refusing to retrain" >&2; exit 2; }
cd "$PROJECT_ROOT"
echo "job=${SLURM_JOB_ID:-manual} host=$(hostname) utc=$(date -u +%FT%TZ) git=$(git rev-parse HEAD)"
nvidia-smi
export DATA_ROOT SIF_PATH THESIS_DEVICE=cuda
# yolo_final_refit.py handles SIGTERM, records an interrupted state, and on a
# later sbatch chooses only the highest valid checkpoint under the same contract.
exec scripts/hpc/run_in_container.sh -- python scripts/benchmark/yolo_final_refit.py train \
  --root "$FINAL_YOLO_ROOT" --darknet "$DARKNET_BIN" --initial-weights "$INITIAL_WEIGHTS" \
  --auto-resume --confirm-training
