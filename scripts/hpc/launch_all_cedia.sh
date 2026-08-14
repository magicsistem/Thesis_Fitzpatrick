#!/usr/bin/env bash
# Submit the DAG only.  It must finish quickly even when invoked by Slurm.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT_CANDIDATE=${PROJECT_ROOT:-}
if [[ -z "$PROJECT_ROOT_CANDIDATE" && -d "$SCRIPT_DIR/../../.git" ]]; then
    PROJECT_ROOT_CANDIDATE=$SCRIPT_DIR/../..
elif [[ -z "$PROJECT_ROOT_CANDIDATE" && -n "${SLURM_SUBMIT_DIR:-}" && "$SLURM_SUBMIT_DIR" != /var/spool/slurm* ]]; then
    PROJECT_ROOT_CANDIDATE=$SLURM_SUBMIT_DIR
fi
[[ -n "$PROJECT_ROOT_CANDIDATE" ]] || { echo "PROJECT_ROOT is required when Slurm executes the launcher from /var/spool/slurm" >&2; exit 2; }
PROJECT_ROOT=$(cd -- "$PROJECT_ROOT_CANDIDATE" && pwd -P)
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/data/raw/isic2018_task1}
SIF_PATH=${SIF_PATH:-${HOME:?}/pytorch_24.01-py3.sif}
MANIFEST_ROOT="$DATA_ROOT/manifests"
TRAIN_MANIFEST=${TRAIN_MANIFEST:-$MANIFEST_ROOT/isic2018_task1_train_disjoint.json}
DEVELOPMENT_MANIFEST=${DEVELOPMENT_MANIFEST:-$MANIFEST_ROOT/isic2018_task1_validation_disjoint.json}
FOLDS_FILE=${FOLDS_FILE:-$MANIFEST_ROOT/isic2018_task1_train_disjoint_folds_5.json}
YOLO_FROZEN_ROOT=${YOLO_FROZEN_ROOT:-$PROJECT_ROOT/results/benchmark_v1/yolo}
BENCHMARK_CONFIG=${BENCHMARK_CONFIG:-$PROJECT_ROOT/.cedia/benchmark.cedia.json}
PIPELINE_BRANCH=${PIPELINE_BRANCH:-fix/hpc-pipeline-validation}
LAUNCH_STARTED=$(date -u +%FT%TZ)
JOB_FILE="$PROJECT_ROOT/.cedia/cedia_job_chain_$(date -u +%Y%m%d_%H%M%S).txt"

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/.cedia"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "$PIPELINE_BRANCH" ]] || { echo "Expected branch $PIPELINE_BRANCH" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || { echo "Git worktree must be clean before submission" >&2; git status --short >&2; exit 2; }
GIT_HEAD=$(git rev-parse HEAD)
git merge-base --is-ancestor "$GIT_HEAD" "origin/$PIPELINE_BRANCH" || { echo "HEAD is not published on origin/$PIPELINE_BRANCH" >&2; exit 2; }

export PROJECT_ROOT DATA_ROOT SIF_PATH TRAIN_MANIFEST DEVELOPMENT_MANIFEST FOLDS_FILE YOLO_FROZEN_ROOT BENCHMARK_CONFIG
export DARKNET_GPU_CONFIRMED=YES
export SMOKE_MANIFEST="$TRAIN_MANIFEST"
registry_tmp=$(mktemp "$PROJECT_ROOT/.cedia/.cedia_job_chain.XXXXXX")
failed=0
finish_registry() {
    local rc=$?
    if ((rc)); then
        printf 'status=failed\nlauncher_finished_utc=%s\n' "$(date -u +%FT%TZ)" >> "$registry_tmp"
        mv -f "$registry_tmp" "$JOB_FILE"
        echo "Pipeline submission failed; registry: $JOB_FILE" >&2
    fi
    exit "$rc"
}
trap finish_registry EXIT
printf 'schema_version=2\nstatus=submitting\nlauncher_started_utc=%s\ngit_head=%s\npipeline_branch=%s\nproject_root=%s\n' \
    "$LAUNCH_STARTED" "$GIT_HEAD" "$PIPELINE_BRANCH" "$PROJECT_ROOT" > "$registry_tmp"

submit() {
    local variable=$1 raw job
    shift
    raw=$(sbatch --parsable "$@")
    job=${raw%%;*}
    [[ "$job" =~ ^[0-9]+$ ]] || { echo "sbatch returned an invalid job id for $variable: $raw" >&2; return 2; }
    printf '%s=%s\n' "$variable" "$job" >> "$registry_tmp"
    printf '%s\n' "$job"
}

PREFLIGHT_JOB=$(submit PREFLIGHT_JOB scripts/hpc/preflight_pipeline_cedia.slurm)
SMOKE_JOB=$(submit SMOKE_JOB --dependency="afterok:$PREFLIGHT_JOB" scripts/hpc/smoke_cedia.slurm)
YOLO_JOB=$(submit YOLO_JOB --dependency="afterok:$SMOKE_JOB" scripts/hpc/train_yolo_cedia.slurm)
FINALIZE_YOLO_JOB=$(submit FINALIZE_YOLO_JOB --dependency="afterok:$YOLO_JOB" scripts/hpc/finalize_yolo_cedia.slurm)
P0_JOB=$(submit P0_JOB --dependency="afterok:$FINALIZE_YOLO_JOB" scripts/hpc/prepare_p0_oof_cedia.slurm)
B2_JOB=$(submit B2_JOB --dependency="afterok:$P0_JOB" scripts/hpc/train_b2_cedia.slurm)
CONFIG_JOB=$(submit CONFIG_JOB --dependency="afterok:$FINALIZE_YOLO_JOB" --job-name=thesis-config --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=60G --gres=gpu:a100-sxm4-40gb:1 --time=00:20:00 --output=slurm-thesis-config-%j.out --error=slurm-thesis-config-%j.err --wrap='cd "$PROJECT_ROOT" && "$PROJECT_ROOT/scripts/hpc/run_in_container.sh" -- python scripts/hpc/configure_benchmark_cedia.py --frozen-yolo-root "$YOLO_FROZEN_ROOT"')
BENCHMARK_JOB=$(submit BENCHMARK_JOB --dependency="afterok:$CONFIG_JOB" scripts/hpc/run_benchmark_cedia.slurm)
B2_FINAL_JOB=$(submit B2_FINAL_JOB --dependency="afterok:$B2_JOB:$CONFIG_JOB" scripts/hpc/finalize_b2_cedia.slurm)

printf 'status=submitted\nlauncher_finished_utc=%s\n' "$(date -u +%FT%TZ)" >> "$registry_tmp"
mv -f "$registry_tmp" "$JOB_FILE"
trap - EXIT
echo "Submitted DAG from $GIT_HEAD"
echo "Job registry: $JOB_FILE"
exit 0
