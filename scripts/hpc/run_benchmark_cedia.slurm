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
echo "job=$SLURM_JOB_ID task=$TASK stage=$STAGE host=$(hostname) utc=$(date -u +%FT%TZ) git=$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} warmup=$WARMUP repetitions=$REPETITIONS"
nvidia-smi
export DATA_ROOT SIF_PATH=${SIF_PATH:-$HOME/pytorch_24.01-py3.sif} THESIS_DEVICE=cuda
cd "$PROJECT_ROOT"
# Keep manifest/idempotency/fold checks and the actual benchmark in one SIF.
# This removes three CPU-only image startups from each benchmark array task.
exec scripts/hpc/run_in_container.sh -- bash -c '
set -Eeuo pipefail
python - "$1" "$2" <<'"'"'PY'"'"'
import json, pathlib, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("split") == "test": raise SystemExit("Full development benchmark refuses sealed test")
if not manifest.get("items"): raise SystemExit("Development manifest is empty")
run = pathlib.Path(sys.argv[2])
if run.is_file():
    status = json.loads(run.read_text(encoding="utf-8")).get("status")
    if status == "completed":
        print("Run already completed and will not be repeated")
        raise SystemExit(0)
    raise SystemExit("Incomplete run exists; inspect it before choosing a new RUN_PREFIX")
print(f"validated dataset={manifest.get('dataset_id')} split={manifest.get('split')} images={len(manifest['items'])}")
PY
python scripts/benchmark/yolov3.py validate-folds --root "$3"
if [[ "$4" == A || "$4" == B1 ]]; then
  exec python scripts/benchmark/run_evaluation.py --evaluation "$4" --dataset isic2018_task1 --split validation \
    --models all --manifest "$1" --data-root "$5" --config "$6" --artifact-root "$7" --run-id "$8" \
    --warmup "$9" --repetitions "${10}" --reuse-p0-cache --confirm-run
fi
exec python scripts/benchmark/run_ablation.py --conditions "$4" --dataset isic2018_task1 --split validation \
  --models all --manifest "$1" --data-root "$5" --config "$6" --artifact-root "$7" --run-id "$8" \
  --warmup "$9" --repetitions "${10}" --reuse-p0-cache --confirm-run
' _ "$DEVELOPMENT_MANIFEST" "$RUN_ROOT/run_manifest.json" "$YOLO_FROZEN_ROOT" "$STAGE" "$DATA_ROOT" \
    "$BENCHMARK_CONFIG" "$ARTIFACT_ROOT" "$RUN_ID" "$WARMUP" "$REPETITIONS"
