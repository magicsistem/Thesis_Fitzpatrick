#!/usr/bin/env bash
# One YOLO fold per GPU.  Exit 75 means an externally interrupted Darknet may
# be resumed once from a hash-verified checkpoint inventory.
#SBATCH --job-name=thesis-yolo
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --time=24:00:00
#SBATCH --array=0-4%2
#SBATCH --signal=B:TERM@120
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -Eeuo pipefail
: "${PROJECT_ROOT:?}"; : "${DATA_ROOT:?}"; : "${SIF_PATH:?}"; : "${DARKNET_GPU_CONFIRMED:?}"
[[ "$DARKNET_GPU_CONFIRMED" == YES ]] || { echo "Darknet GPU build has not been explicitly confirmed" >&2; exit 2; }
PROJECT_ROOT=$(realpath "$PROJECT_ROOT"); DATA_ROOT=$(realpath "$DATA_ROOT"); SIF_PATH=$(realpath "$SIF_PATH")
TRAIN_MANIFEST=${TRAIN_MANIFEST:-$DATA_ROOT/manifests/isic2018_task1_train_disjoint.json}
FOLDS_FILE=${FOLDS_FILE:-$DATA_ROOT/manifests/isic2018_task1_train_disjoint_folds_5.json}
YOLO_ROOT=${YOLO_ROOT:-$PROJECT_ROOT/results/benchmark_v1/yolo}
DARKNET_BIN=${DARKNET_BIN:-$PROJECT_ROOT/models/yolov3-darknet/source/darknet}
INITIAL_WEIGHTS=${INITIAL_WEIGHTS:-$PROJECT_ROOT/models/yolov3-darknet/checkpoints/darknet53.conv.74}
FOLD=${SLURM_ARRAY_TASK_ID:?This template must run as an array task}
FOLD_DIR=$YOLO_ROOT/fold-$FOLD
MAX_ATTEMPTS=${YOLO_MAX_ATTEMPTS:-2}
[[ "$MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ && "$MAX_ATTEMPTS" -le 3 ]] || { echo "YOLO_MAX_ATTEMPTS must be 1..3" >&2; exit 2; }

for path in "$PROJECT_ROOT" "$DATA_ROOT" "$SIF_PATH" "$TRAIN_MANIFEST" "$FOLDS_FILE" "$DARKNET_BIN" "$INITIAL_WEIGHTS" "$FOLD_DIR/lesion.data" "$FOLD_DIR/lesion-yolov3.cfg" "$FOLD_DIR/label_audit.json"; do
    [[ -e "$path" ]] || { echo "Missing prerequisite: $path" >&2; exit 2; }
done
[[ ! -e "$FOLD_DIR/frozen.json" ]] || { echo "Fold $FOLD is already frozen; refusing to retrain" >&2; exit 2; }
mkdir -p "$FOLD_DIR/backup" "$FOLD_DIR/training" "$PROJECT_ROOT/.cedia/diagnostics"
cd "$PROJECT_ROOT"
if ! scripts/hpc/run_in_container.sh --check-runtime-paths; then
    echo "Permanent container-directory configuration error; Darknet was not started and no YOLO attempt was consumed." >&2
    exit 2
fi
source scripts/hpc/runtime_diagnostics.sh
echo "job=${SLURM_JOB_ID:-unset} array_job=${SLURM_ARRAY_JOB_ID:-manual} task=$FOLD fold=$FOLD"
echo "host=$(hostname) utc=$(date -u +%FT%TZ) git_head=$(git rev-parse HEAD) max_attempts=$MAX_ATTEMPTS"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

scripts/hpc/run_in_container.sh --cpu -- python - "$TRAIN_MANIFEST" "$FOLDS_FILE" "$FOLD" <<'PY'
import json, sys
manifest, folds_path, fold = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(manifest, encoding="utf-8") as stream: payload = json.load(stream)
with open(folds_path, encoding="utf-8") as stream: folds = json.load(stream)
entry = next((item for item in folds.get("folds", []) if item.get("fold") == fold), None)
if payload.get("split") != "train" or entry is None or set(entry["train_ids"]) & set(entry["validation_ids"]): raise SystemExit("Invalid or leaking fold")
print(f"validated train manifest and fold {fold}")
PY

child=""; child_pgid=""
stop_child() {
    local signal_name=$1
    runtime_diag_note_signal "$signal_name"
    echo "signal received at $(date -u +%FT%TZ); preserving validated Darknet backups"
    [[ -z "$child_pgid" ]] || kill -TERM -- "-$child_pgid" 2>/dev/null || true
    [[ -z "$child" ]] || wait "$child" 2>/dev/null || true
    runtime_diag_stop
    exit 143
}
trap 'stop_child TERM' TERM INT USR1
trap 'runtime_diag_stop' EXIT
export DATA_ROOT SIF_PATH

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    echo "yolo_attempt=$attempt/$MAX_ATTEMPTS utc=$(date -u +%FT%TZ)"
    RUNTIME_DIAGNOSTICS_LOG="$PROJECT_ROOT/.cedia/diagnostics/yolo-${SLURM_JOB_ID:-manual}-fold-$FOLD-attempt-$attempt.log"
    export RUNTIME_DIAGNOSTICS_LOG
    setsid "$PROJECT_ROOT/scripts/hpc/run_in_container.sh" -- python scripts/benchmark/yolov3.py train \
        --darknet "$DARKNET_BIN" --data "$FOLD_DIR/lesion.data" --cfg "$FOLD_DIR/lesion-yolov3.cfg" \
        --initial-weights "$INITIAL_WEIGHTS" --fold "$FOLD" --output "$FOLD_DIR/training" --confirm-training &
    child=$!; child_pgid=$child
    set +e; wait "$child"; rc=$?; set -e
    child=""; child_pgid=""; unset RUNTIME_DIAGNOSTICS_LOG
    echo "finished fold=$FOLD attempt=$attempt returncode=$rc utc=$(date -u +%FT%TZ)"
    [[ "$rc" != 75 ]] && exit "$rc"
    (( attempt < MAX_ATTEMPTS )) || exit 75
    echo "retrying only because Python recorded an interrupted state and validated a checkpoint"
done
