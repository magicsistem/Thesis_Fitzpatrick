#!/usr/bin/env bash
# Freeze exactly 75 completed B2 checkpoints, then evaluate B2 on development validation.
#SBATCH --job-name=thesis-b2-finalize
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail
trap 'rc=$?; echo "B2 finalization failed rc=$rc line=$LINENO" >&2; exit "$rc"' ERR
: "${PROJECT_ROOT:?}"; : "${DATA_ROOT:?}"; : "${DEVELOPMENT_MANIFEST:?}"; : "${BENCHMARK_CONFIG:?}"; : "${YOLO_FROZEN_ROOT:?}"
PROJECT_ROOT=$(realpath "$PROJECT_ROOT"); DATA_ROOT=$(realpath "$DATA_ROOT")
B2_ROOT=${B2_ROOT:-$PROJECT_ROOT/results/benchmark_v1/b2}
B2_REGISTRY=${B2_REGISTRY:-$PROJECT_ROOT/results/benchmark_v1/b2_frozen.json}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-$PROJECT_ROOT/results/benchmark_v1}
RUN_ID=${RUN_ID:-cedia-development-b2}
export DATA_ROOT SIF_PATH=${SIF_PATH:-$HOME/pytorch_24.01-py3.sif}
cd "$PROJECT_ROOT"
echo "job=$SLURM_JOB_ID host=$(hostname) utc=$(date -u +%FT%TZ) git=$(git rev-parse HEAD)"
nvidia-smi
exec scripts/hpc/run_in_container.sh -- bash -c '
set -Eeuo pipefail
python scripts/benchmark/freeze_b2.py --training-root "$1" --base-config "$2" \
  --dataset isic2018_task1 --output "$3" --note "Five-fold standardized B2 completed on CEDIA"
python scripts/benchmark/yolov3.py validate-folds --root "$8"
run_manifest="$6/runs/$7/run_manifest.json"
if [[ -f "$run_manifest" ]]; then
  [[ "$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get(\"status\"))" "$run_manifest")" == completed ]] || { echo "Existing B2 evaluation is incomplete: $run_manifest" >&2; exit 2; }
  echo "B2 registry and evaluation already completed and validated"
  exit 0
fi
python scripts/benchmark/run_evaluation.py --evaluation B2 --dataset isic2018_task1 --split validation \
  --models all --manifest "$4" --data-root "$5" --config "$2" --b2-config "$3" \
  --artifact-root "$6" --run-id "$7" --warmup 1 --repetitions 3 --confirm-run
' _ "$B2_ROOT" "$BENCHMARK_CONFIG" "$B2_REGISTRY" "$DEVELOPMENT_MANIFEST" "$DATA_ROOT" "$ARTIFACT_ROOT" "$RUN_ID" "$YOLO_FROZEN_ROOT"
