#!/usr/bin/env bash
# Repair/finalize one existing A/B1 run without repeating valid combinations.
#SBATCH --job-name=thesis-eval-repair
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail
trap 'rc=$?; echo "evaluation repair failed rc=$rc line=$LINENO utc=$(date -u +%FT%TZ)" >&2; exit "$rc"' ERR

: "${PROJECT_ROOT:?Export PROJECT_ROOT before sbatch}"
: "${DATA_ROOT:?Export DATA_ROOT before sbatch}"
: "${DEVELOPMENT_MANIFEST:?Export DEVELOPMENT_MANIFEST before sbatch}"
: "${TARGET_RUN_ID:?Export TARGET_RUN_ID, e.g. cedia-development-a-23188}"

PROJECT_ROOT=$(realpath "$PROJECT_ROOT")
DATA_ROOT=$(realpath "$DATA_ROOT")
DEVELOPMENT_MANIFEST=$(realpath "$DEVELOPMENT_MANIFEST")
BENCHMARK_CONFIG=${BENCHMARK_CONFIG:-$PROJECT_ROOT/.cedia/benchmark.cedia.json}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-$PROJECT_ROOT/results/benchmark_v1}
TARGET_RUN="$ARTIFACT_ROOT/runs/$TARGET_RUN_ID"
WARMUP=${WARMUP:-1}
REPETITIONS=${REPETITIONS:-3}

for path in "$PROJECT_ROOT" "$DATA_ROOT" "$DEVELOPMENT_MANIFEST" "$BENCHMARK_CONFIG" "$TARGET_RUN/run_manifest.json"; do
    [[ -e "$path" ]] || { echo "Missing prerequisite: $path" >&2; exit 2; }
done

echo "job=$SLURM_JOB_ID target=$TARGET_RUN_ID host=$(hostname) utc=$(date -u +%FT%TZ) git=$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
nvidia-smi
export DATA_ROOT SIF_PATH=${SIF_PATH:-$HOME/pytorch_24.01-py3.sif} THESIS_DEVICE=cuda
cd "$PROJECT_ROOT"

exec scripts/hpc/run_in_container.sh -- python scripts/benchmark/resume_evaluation.py \
    --run-directory "$TARGET_RUN" \
    --manifest "$DEVELOPMENT_MANIFEST" \
    --data-root "$DATA_ROOT" \
    --config "$BENCHMARK_CONFIG" \
    --warmup "$WARMUP" \
    --repetitions "$REPETITIONS" \
    --reuse-p0-cache \
    --confirm-repair
