#!/usr/bin/env bash
# CEDIA Open OnDemand submission template; S01-S15 x five folds, one GPU each.
#SBATCH --job-name=thesis-b2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --time=24:00:00
#SBATCH --array=0-74%2
#SBATCH --signal=B:TERM@120
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail
: "${PROJECT_ROOT:?Export PROJECT_ROOT before submission}"
: "${DATA_ROOT:?Export DATA_ROOT before submission}"
: "${SIF_PATH:?Export the SIF_PATH observed by the CEDIA diagnostic}"
: "${YOLO_FROZEN_ROOT:?Export the directory containing fold-0/frozen.json through fold-4/frozen.json}"

PROJECT_ROOT=$(realpath "$PROJECT_ROOT")
DATA_ROOT=$(realpath "$DATA_ROOT")
SIF_PATH=$(realpath "$SIF_PATH")
YOLO_FROZEN_ROOT=$(realpath "$YOLO_FROZEN_ROOT")
TRAIN_MANIFEST=${TRAIN_MANIFEST:-$DATA_ROOT/manifests/isic2018_task1_train.json}
FOLDS_FILE=${FOLDS_FILE:-$DATA_ROOT/manifests/isic2018_task1_train_folds_5.json}
P0_ROOT=${P0_ROOT:-$PROJECT_ROOT/results/benchmark_v1/preprocessing}
B2_ROOT=${B2_ROOT:-$PROJECT_ROOT/results/benchmark_v1/b2}
EPOCHS=${EPOCHS:-100}
SEED=${SEED:-20260806}
TASK=${SLURM_ARRAY_TASK_ID:?This template must run as an array task}
methods=(S01 S02 S03 S04 S05 S06 S07 S08 S09 S10 S11 S12 S13 S14 S15)
METHOD_INDEX=$((TASK / 5))
FOLD=$((TASK % 5))
METHOD_ID=${methods[$METHOD_INDEX]}
OUTPUT=$B2_ROOT/$METHOD_ID/fold-$FOLD

for path in "$PROJECT_ROOT" "$DATA_ROOT" "$SIF_PATH" "$TRAIN_MANIFEST" "$FOLDS_FILE" "$P0_ROOT" "$YOLO_FROZEN_ROOT"; do
    [[ -e "$path" ]] || { echo "Missing prerequisite: $path" >&2; exit 2; }
done
[[ -f "$P0_ROOT/oof_manifest.json" ]] || { echo "B2 requires completed leakage-safe OOF P0: $P0_ROOT/oof_manifest.json" >&2; exit 2; }

python_bin=${PYTHON_BIN:-$PROJECT_ROOT/.cedia/venv/bin/python}
if [[ -n "${B2_VENV_ROOT:-}" && -x "$B2_VENV_ROOT/$METHOD_ID/bin/python" ]]; then
    python_bin=$B2_VENV_ROOT/$METHOD_ID/bin/python
elif [[ -n "${VENV_PATH:-}" ]]; then
    [[ -x "$VENV_PATH/bin/python" ]] || { echo "VENV_PATH has no executable bin/python" >&2; exit 2; }
    python_bin=$VENV_PATH/bin/python
fi

mkdir -p "$OUTPUT" "$B2_ROOT/logs"
log=$B2_ROOT/logs/b2-${SLURM_ARRAY_JOB_ID:-manual}-$TASK-$METHOD_ID-fold$FOLD.log
exec > >(tee -a "$log") 2>&1
echo "job=${SLURM_JOB_ID:-unset} array_job=${SLURM_ARRAY_JOB_ID:-unset} task=$TASK method=$METHOD_ID fold=$FOLD"
echo "host=$(hostname) utc=$(date -u +%FT%TZ) python=$python_bin git_head=$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

python3 - "$TRAIN_MANIFEST" "$FOLDS_FILE" "$YOLO_FROZEN_ROOT" "$P0_ROOT/oof_manifest.json" <<'PY'
import hashlib, json, pathlib, sys

manifest_path, folds_path, frozen_root, p0_manifest_path = map(pathlib.Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
folds = json.loads(folds_path.read_text(encoding="utf-8"))
if manifest.get("split") != "train": raise SystemExit("B2 requires split=train; sealed test is forbidden")
if len(folds.get("folds", [])) != 5: raise SystemExit("B2 requires exactly five folds")
if json.loads(p0_manifest_path.read_text(encoding="utf-8")).get("status") != "completed": raise SystemExit("B2 requires completed leakage-safe OOF P0")
for fold in folds["folds"]:
    if set(fold["train_ids"]) & set(fold["validation_ids"]): raise SystemExit("Fold leakage detected")
for index in range(5):
    path = frozen_root / f"fold-{index}" / "frozen.json"
    if not path.is_file(): raise SystemExit(f"Missing frozen YOLO manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen" or not payload.get("identity_hash"):
        raise SystemExit(f"YOLO fold is not frozen: {path}")
    for name in ("cfg", "weights"):
        artifact = pathlib.Path(payload[f"{name}_path"])
        if not artifact.is_file(): raise SystemExit(f"Missing frozen {name}: {artifact}")
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if digest != payload[f"{name}_sha256"]: raise SystemExit(f"Modified frozen {name}: {artifact}")
print("validated train-only folds and five frozen YOLO detectors")
PY

resume=()
[[ ! -f "$OUTPUT/training_state.pt" ]] || resume=(--resume)
child=""
stop_child() {
    echo "signal received at $(date -u +%FT%TZ); last completed epoch remains resumable"
    [[ -z "$child" ]] || kill -TERM "$child" 2>/dev/null || true
    [[ -z "$child" ]] || wait "$child" 2>/dev/null || true
    exit 143
}
trap stop_child TERM INT
export DATA_ROOT SIF_PATH VENV_PATH=${VENV_PATH:-$PROJECT_ROOT/.cedia/venv}
"$PROJECT_ROOT/scripts/hpc/run_in_container.sh" -- "$python_bin" scripts/benchmark/train_backend.py \
    --method "$METHOD_ID" --manifest "$TRAIN_MANIFEST" --data-root "$DATA_ROOT" \
    --folds "$FOLDS_FILE" --fold "$FOLD" --p0-root "$P0_ROOT" --output "$OUTPUT" \
    --epochs "$EPOCHS" --seed "$SEED" --device cuda "${resume[@]}" --confirm-training &
child=$!
set +e; wait "$child"; rc=$?; set -e
echo "finished method=$METHOD_ID fold=$FOLD returncode=$rc utc=$(date -u +%FT%TZ)"
exit "$rc"
