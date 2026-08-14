#!/usr/bin/env bash
# Six independent full-development stages: A, B1, C0, C1, C2 and C3.
#SBATCH --job-name=thesis-benchmark
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --time=48:00:00
#SBATCH --array=0-5%1
#SBATCH --signal=B:TERM@180
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -Eeuo pipefail
trap 'rc=$?; echo "benchmark failed rc=$rc line=$LINENO utc=$(date -u +%FT%TZ)" >&2; exit "$rc"' ERR
: "${PROJECT_ROOT:?Export PROJECT_ROOT before sbatch}"
: "${DATA_ROOT:?Export DATA_ROOT before sbatch}"
: "${DEVELOPMENT_MANIFEST:?Export the complete non-test development manifest}"
: "${YOLO_FROZEN_ROOT:?Export the five-fold frozen YOLO root}"
PROJECT_ROOT=$(realpath "$PROJECT_ROOT")
DATA_ROOT=$(realpath "$DATA_ROOT")
DEVELOPMENT_MANIFEST=$(realpath "$DEVELOPMENT_MANIFEST")
BENCHMARK_CONFIG=${BENCHMARK_CONFIG:-$PROJECT_ROOT/configs/benchmark/default.json}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-$PROJECT_ROOT/results/benchmark_v1}
WARMUP=${WARMUP:-1}
REPETITIONS=${REPETITIONS:-3}
TASK=${SLURM_ARRAY_TASK_ID:?Submit this file as its declared array}
stages=(A B1 C0 C1 C2 C3)
STAGE=${stages[$TASK]}
RUN_ID=${RUN_PREFIX:-cedia-development}-${STAGE,,}-${SLURM_ARRAY_JOB_ID:-manual}
RUN_ROOT=$ARTIFACT_ROOT/runs/$RUN_ID

for path in "$PROJECT_ROOT" "$DATA_ROOT" "$DEVELOPMENT_MANIFEST" "$BENCHMARK_CONFIG" "$PROJECT_ROOT/.cedia/venv/bin/python"; do
    [[ -e "$path" ]] || { echo "Missing prerequisite: $path" >&2; exit 2; }
done
python3 - "$DEVELOPMENT_MANIFEST" <<'PY'
import json, pathlib, sys
p = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if p.get("split") == "test": raise SystemExit("Full development benchmark refuses sealed test")
if not p.get("items"): raise SystemExit("Development manifest is empty")
print(f"validated dataset={p.get('dataset_id')} split={p.get('split')} images={len(p['items'])}")
PY
if [[ -f "$RUN_ROOT/run_manifest.json" ]]; then
    state=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$RUN_ROOT/run_manifest.json")
    [[ "$state" == completed ]] && { echo "Run already completed and will not be repeated: $RUN_ID"; exit 0; }
    echo "Incomplete run exists at $RUN_ROOT; inspect it before choosing a new RUN_PREFIX" >&2
    exit 2
fi

mkdir -p "$ARTIFACT_ROOT/logs"
exec > >(tee -a "$ARTIFACT_ROOT/logs/benchmark-${SLURM_ARRAY_JOB_ID:-manual}-$TASK-$STAGE.log") 2>&1
echo "job=$SLURM_JOB_ID task=$TASK stage=$STAGE host=$(hostname) utc=$(date -u +%FT%TZ) git=$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} warmup=$WARMUP repetitions=$REPETITIONS"
nvidia-smi
export DATA_ROOT SIF_PATH=${SIF_PATH:-$HOME/pytorch_24.01-py3.sif} THESIS_DEVICE=cuda
cd "$PROJECT_ROOT"
scripts/hpc/run_in_container.sh -- python scripts/benchmark/yolov3.py validate-folds --root "$YOLO_FROZEN_ROOT"
if [[ "$STAGE" == A || "$STAGE" == B1 ]]; then
    command=(python scripts/benchmark/run_evaluation.py --evaluation "$STAGE" --dataset isic2018_task1 --split validation
        --models all --manifest "$DEVELOPMENT_MANIFEST" --data-root "$DATA_ROOT" --config "$BENCHMARK_CONFIG"
        --artifact-root "$ARTIFACT_ROOT" --run-id "$RUN_ID" --warmup "$WARMUP" --repetitions "$REPETITIONS" --reuse-p0-cache --confirm-run)
else
    command=(python scripts/benchmark/run_ablation.py --conditions "$STAGE" --dataset isic2018_task1 --split validation
        --models all --manifest "$DEVELOPMENT_MANIFEST" --data-root "$DATA_ROOT" --config "$BENCHMARK_CONFIG"
        --artifact-root "$ARTIFACT_ROOT" --run-id "$RUN_ID" --warmup "$WARMUP" --repetitions "$REPETITIONS" --reuse-p0-cache --confirm-run)
fi
exec scripts/hpc/run_in_container.sh -- "${command[@]}"
