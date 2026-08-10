#!/usr/bin/env bash
# CEDIA Open OnDemand submission template; one YOLO fold per single-GPU task.
#SBATCH --job-name=thesis-yolo
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --time=24:00:00
#SBATCH --array=0-4%2
#SBATCH --signal=B:TERM@120
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail
: "${PROJECT_ROOT:?Export PROJECT_ROOT before submission}"
: "${DATA_ROOT:?Export DATA_ROOT before submission}"
: "${SIF_PATH:?Export the SIF_PATH observed by the CEDIA diagnostic}"
: "${DARKNET_GPU_CONFIRMED:?Set DARKNET_GPU_CONFIRMED=YES only after validating the CEDIA GPU build}"
[[ "$DARKNET_GPU_CONFIRMED" == YES ]] || { echo "Darknet GPU build has not been explicitly confirmed" >&2; exit 2; }

PROJECT_ROOT=$(realpath "$PROJECT_ROOT")
DATA_ROOT=$(realpath "$DATA_ROOT")
SIF_PATH=$(realpath "$SIF_PATH")
TRAIN_MANIFEST=${TRAIN_MANIFEST:-$DATA_ROOT/manifests/isic2018_task1_train_disjoint.json}
FOLDS_FILE=${FOLDS_FILE:-$DATA_ROOT/manifests/isic2018_task1_train_disjoint_folds_5.json}
YOLO_ROOT=${YOLO_ROOT:-$PROJECT_ROOT/results/benchmark_v1/yolo}
DARKNET_BIN=${DARKNET_BIN:-$PROJECT_ROOT/models/yolov3-darknet/source/darknet}
INITIAL_WEIGHTS=${INITIAL_WEIGHTS:-$PROJECT_ROOT/models/yolov3-darknet/checkpoints/darknet53.conv.74}
PYTHON_BIN=${PYTHON_BIN:-python3}
FOLD=${SLURM_ARRAY_TASK_ID:?This template must run as an array task}
FOLD_DIR=$YOLO_ROOT/fold-$FOLD

for path in "$PROJECT_ROOT" "$DATA_ROOT" "$SIF_PATH" "$TRAIN_MANIFEST" "$FOLDS_FILE" "$DARKNET_BIN" "$INITIAL_WEIGHTS" "$FOLD_DIR/lesion.data" "$FOLD_DIR/lesion-yolov3.cfg" "$FOLD_DIR/label_audit.json"; do
    [[ -e "$path" ]] || { echo "Missing prerequisite: $path" >&2; exit 2; }
done
[[ ! -e "$FOLD_DIR/frozen.json" ]] || { echo "Fold $FOLD is already frozen; refusing to retrain" >&2; exit 2; }

mkdir -p "$FOLD_DIR/training" "$YOLO_ROOT/logs"
log=$YOLO_ROOT/logs/yolo-${SLURM_ARRAY_JOB_ID:-manual}-$FOLD.log
exec > >(tee -a "$log") 2>&1
echo "job=${SLURM_JOB_ID:-unset} array_job=${SLURM_ARRAY_JOB_ID:-unset} task=$FOLD fold=$FOLD"
echo "host=$(hostname) utc=$(date -u +%FT%TZ) git_head=$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

python3 - "$TRAIN_MANIFEST" "$FOLDS_FILE" "$FOLD" <<'PY'
import json, sys
manifest, folds_path, fold = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(manifest, encoding="utf-8") as stream: payload = json.load(stream)
with open(folds_path, encoding="utf-8") as stream: folds = json.load(stream)
if payload.get("split") != "train": raise SystemExit("YOLO training requires split=train")
entry = next((item for item in folds.get("folds", []) if item.get("fold") == fold), None)
if entry is None or set(entry["train_ids"]) & set(entry["validation_ids"]): raise SystemExit("Invalid or leaking fold")
print(f"validated train manifest and fold {fold}")
PY

resume_args=()
if [[ -n "${RESUME_WEIGHTS:-}" ]]; then
    [[ -f "$RESUME_WEIGHTS" ]] || { echo "RESUME_WEIGHTS does not exist" >&2; exit 2; }
    resume_args=(--resume "$RESUME_WEIGHTS")
elif [[ -d "$FOLD_DIR/backup" ]]; then
    latest=$(find "$FOLD_DIR/backup" -maxdepth 1 -type f -name '*.weights' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)
    [[ -z "$latest" ]] || resume_args=(--resume "$latest")
fi

child=""
stop_child() {
    echo "signal received at $(date -u +%FT%TZ); Darknet backups remain available for resubmission"
    [[ -z "$child" ]] || kill -TERM "$child" 2>/dev/null || true
    [[ -z "$child" ]] || wait "$child" 2>/dev/null || true
    exit 143
}
trap stop_child TERM INT
export DATA_ROOT SIF_PATH
"$PROJECT_ROOT/scripts/hpc/run_in_container.sh" -- "$PYTHON_BIN" scripts/benchmark/yolov3.py train \
    --darknet "$DARKNET_BIN" --data "$FOLD_DIR/lesion.data" \
    --cfg "$FOLD_DIR/lesion-yolov3.cfg" --initial-weights "$INITIAL_WEIGHTS" \
    --output "$FOLD_DIR/training" "${resume_args[@]}" --confirm-training &
child=$!
set +e; wait "$child"; rc=$?; set -e
echo "finished fold=$FOLD returncode=$rc utc=$(date -u +%FT%TZ)"
exit "$rc"
