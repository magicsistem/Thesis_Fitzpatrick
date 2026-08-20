#!/usr/bin/env bash
#SBATCH --job-name=thesis-post05-ablate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --time=48:00:00
#SBATCH --array=0-4%2
#SBATCH --signal=B:TERM@180
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -Eeuo pipefail
trap 'rc=$?; echo "post05 ablation failed rc=$rc line=$LINENO utc=$(date -u +%FT%TZ)" >&2; exit "$rc"' ERR
: "${PROJECT_ROOT:?Export PROJECT_ROOT}"
: "${DATA_ROOT:?Export DATA_ROOT}"
: "${SIF_PATH:?Export SIF_PATH}"
: "${DEVELOPMENT_MANIFEST:?Export the non-test validation_disjoint manifest}"
PROJECT_ROOT=$(realpath "$PROJECT_ROOT")
DATA_ROOT=$(realpath "$DATA_ROOT")
SIF_PATH=$(realpath "$SIF_PATH")
DEVELOPMENT_MANIFEST=$(realpath "$DEVELOPMENT_MANIFEST")
FINAL_B1_CONFIG=${FINAL_B1_CONFIG:-$PROJECT_ROOT/.cedia/post05/benchmark.final_b1.json}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-$PROJECT_ROOT/results/benchmark_v1}
WARMUP=${WARMUP:-1}
REPETITIONS=${REPETITIONS:-3}
METHODS=${METHODS:-all}
TASK=${SLURM_ARRAY_TASK_ID:?Submit as the declared array}
conditions=(B1_FULL B1_NO_HAIR B1_NO_YOLO B1_NO_FOV B1_NO_POST)
CONDITION=${conditions[$TASK]}
RUN_PREFIX=${RUN_PREFIX:-post05-ablation-${SLURM_ARRAY_JOB_ID:-manual}}
RUN_ID=${RUN_PREFIX}-${CONDITION,,}
for path in "$PROJECT_ROOT" "$DATA_ROOT" "$SIF_PATH" "$DEVELOPMENT_MANIFEST" "$FINAL_B1_CONFIG"; do
  [[ -e "$path" ]] || { echo "Missing prerequisite: $path" >&2; exit 2; }
done
cd "$PROJECT_ROOT"
export DATA_ROOT SIF_PATH THESIS_DEVICE=cuda
echo "condition=$CONDITION run_id=$RUN_ID host=$(hostname) utc=$(date -u +%FT%TZ)"
exec scripts/hpc/run_in_container.sh -- python scripts/benchmark/run_component_ablation.py \
  --conditions "$CONDITION" --dataset isic2018_task1 --split validation --models "$METHODS" \
  --manifest "$DEVELOPMENT_MANIFEST" --data-root "$DATA_ROOT" --config "$FINAL_B1_CONFIG" \
  --artifact-root "$ARTIFACT_ROOT" --run-id "$RUN_ID" --warmup "$WARMUP" --repetitions "$REPETITIONS" \
  --reuse-p0-cache --confirm-run
