#!/usr/bin/env bash
# Submit only the resumable production DAG; it never submits heavy preflight.
set -Eeuo pipefail

usage() { echo "Usage: scripts/hpc/launch_pipeline_cedia.sh --resume-existing --skip-preflight" >&2; }
[[ "${1:-}" == --resume-existing && "${2:-}" == --skip-preflight && $# == 2 ]] || { usage; exit 2; }
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd -P)}
PROJECT_ROOT=$(cd -- "$PROJECT_ROOT" && pwd -P)
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/data/raw/isic2018_task1}
SIF_PATH=${SIF_PATH:-${HOME:?}/pytorch_24.01-py3.sif}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-$DATA_ROOT/manifests/isic2018_task1_train_disjoint.json}
DEVELOPMENT_MANIFEST=${DEVELOPMENT_MANIFEST:-$DATA_ROOT/manifests/isic2018_task1_validation_disjoint.json}
FOLDS_FILE=${FOLDS_FILE:-$DATA_ROOT/manifests/isic2018_task1_train_disjoint_folds_5.json}
YOLO_ROOT=${YOLO_ROOT:-$PROJECT_ROOT/results/benchmark_v1/yolo}
YOLO_FROZEN_ROOT=${YOLO_FROZEN_ROOT:-$YOLO_ROOT}
P0_ROOT=${P0_ROOT:-$PROJECT_ROOT/results/benchmark_v1/preprocessing}
B2_ROOT=${B2_ROOT:-$PROJECT_ROOT/results/benchmark_v1/b2}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-$PROJECT_ROOT/results/benchmark_v1}
BENCHMARK_CONFIG=${BENCHMARK_CONFIG:-$PROJECT_ROOT/.cedia/benchmark.cedia.json}
PIPELINE_BRANCH=${PIPELINE_BRANCH:-fix/hpc-pipeline-validation}
YOLO_ARRAY=0,2,3,4%2
B2_ARRAY=0-74%2

cd "$PROJECT_ROOT"
mkdir -p .cedia
[[ "$(git rev-parse --abbrev-ref HEAD)" == "$PIPELINE_BRANCH" ]] || { echo "Expected branch $PIPELINE_BRANCH" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || { echo "Git worktree must be clean before submission" >&2; exit 2; }
GIT_HEAD=$(git rev-parse HEAD)
git merge-base --is-ancestor "$GIT_HEAD" "origin/$PIPELINE_BRANCH" || { echo "HEAD is not published on origin/$PIPELINE_BRANCH" >&2; exit 2; }
for path in "$DATA_ROOT" "$SIF_PATH" "$TRAIN_MANIFEST" "$DEVELOPMENT_MANIFEST" "$FOLDS_FILE" "$YOLO_ROOT"; do
    [[ -e "$path" ]] || { echo "Missing pipeline prerequisite: $path" >&2; exit 2; }
done
for marker in "$P0_ROOT/oof_manifest.json" "$ARTIFACT_ROOT/b2_frozen.json" "$ARTIFACT_ROOT/runs"/*/run_manifest.json; do
    [[ ! -e "$marker" ]] || { echo "Existing downstream result requires inspection; refusing a duplicate resume launch: $marker" >&2; exit 2; }
done
command -v squeue >/dev/null 2>&1 || { echo "squeue is required to reject an active duplicate launch" >&2; exit 2; }
for manifest in .cedia/resume_pipeline_*.json; do
    [[ -e "$manifest" ]] || continue
    while IFS= read -r job; do
        [[ -z "$job" ]] || ! squeue -h -j "$job" -o '%A' 2>/dev/null | grep -q . || { echo "Active resume job $job from $manifest; refusing duplicate launch" >&2; exit 2; }
    done < <(sed -n 's/.*"job_id":"\([0-9][0-9]*\)".*/\1/p' "$manifest")
done

export PROJECT_ROOT DATA_ROOT SIF_PATH TRAIN_MANIFEST DEVELOPMENT_MANIFEST FOLDS_FILE YOLO_ROOT YOLO_FROZEN_ROOT P0_ROOT B2_ROOT ARTIFACT_ROOT BENCHMARK_CONFIG
export DARKNET_GPU_CONFIRMED=YES
submit() {
    local name=$1 dependency=$2 raw job
    shift 2
    if [[ "$dependency" == none ]]; then raw=$(sbatch --parsable --export=ALL "$@")
    else raw=$(sbatch --parsable --export=ALL --dependency="afterok:$dependency" "$@")
    fi
    job=${raw%%;*}
    [[ "$job" =~ ^[0-9]+$ ]] || { echo "Invalid sbatch job id for $name: $raw" >&2; return 2; }
    printf 'submitted stage=%s job_id=%s dependency=%s\n' "$name" "$job" "$dependency" >&2
    printf '%s\n' "$job"
}

VALIDATE_JOB=$(submit validation none scripts/hpc/validate_existing_cedia.slurm)
YOLO_JOB=$(submit yolo "$VALIDATE_JOB" --array="$YOLO_ARRAY" scripts/hpc/train_yolo_cedia.slurm)
FINALIZE_YOLO_JOB=$(submit finalize_yolo "$YOLO_JOB" --array=0-4%2 scripts/hpc/finalize_yolo_cedia.slurm)
P0_JOB=$(submit p0 "$FINALIZE_YOLO_JOB" scripts/hpc/prepare_p0_oof_cedia.slurm)
B2_JOB=$(submit b2 "$P0_JOB" --array="$B2_ARRAY" scripts/hpc/train_b2_cedia.slurm)
CONFIG_JOB=$(submit config "$B2_JOB" --job-name=thesis-config --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=60G --gres=gpu:a100-sxm4-40gb:1 --time=00:20:00 --output=slurm-thesis-config-%j.out --error=slurm-thesis-config-%j.err --wrap "cd \"$PROJECT_ROOT\" && \"$PROJECT_ROOT/scripts/hpc/run_in_container.sh\" -- python scripts/hpc/configure_benchmark_cedia.py --frozen-yolo-root \"$YOLO_FROZEN_ROOT\"")
BENCHMARK_JOB=$(submit benchmark "$CONFIG_JOB" scripts/hpc/run_benchmark_cedia.slurm)
B2_FINAL_JOB=$(submit finalize_b2 "$BENCHMARK_JOB" scripts/hpc/finalize_b2_cedia.slurm)

MANIFEST="$PROJECT_ROOT/.cedia/resume_pipeline_$(date -u +%Y%m%d_%H%M%S).json"
MANIFEST_TMP=$(mktemp "$PROJECT_ROOT/.cedia/.resume_pipeline.XXXXXX")
cat > "$MANIFEST_TMP" <<EOF
{"schema_version":1,"status":"submitted","created_utc":"$(date -u +%FT%TZ)","git_sha":"$GIT_HEAD","skip_preflight":true,"fold_1_reused":true,"yolo_folds":"0,2,3,4","b2_tasks":75,"paths":{"project_root":"$PROJECT_ROOT","data_root":"$DATA_ROOT","sif_path":"$SIF_PATH","yolo_root":"$YOLO_ROOT"},"jobs":[{"stage":"validation","job_id":"$VALIDATE_JOB","dependency":null},{"stage":"yolo","job_id":"$YOLO_JOB","dependency":"afterok:$VALIDATE_JOB","array":"$YOLO_ARRAY"},{"stage":"finalize_yolo","job_id":"$FINALIZE_YOLO_JOB","dependency":"afterok:$YOLO_JOB","array":"0-4%2"},{"stage":"p0","job_id":"$P0_JOB","dependency":"afterok:$FINALIZE_YOLO_JOB"},{"stage":"b2","job_id":"$B2_JOB","dependency":"afterok:$P0_JOB","array":"$B2_ARRAY"},{"stage":"config","job_id":"$CONFIG_JOB","dependency":"afterok:$B2_JOB"},{"stage":"benchmark","job_id":"$BENCHMARK_JOB","dependency":"afterok:$CONFIG_JOB"},{"stage":"finalize_b2","job_id":"$B2_FINAL_JOB","dependency":"afterok:$BENCHMARK_JOB"}]}
EOF
mv -f "$MANIFEST_TMP" "$MANIFEST"
printf '%s\n' "Resume DAG submitted from $GIT_HEAD"
printf '%s\n' "preflight=NOT_SUBMITTED fold_1=reused yolo_array=$YOLO_ARRAY b2_array=$B2_ARRAY"
printf '%s\n' "DAG: validation($VALIDATE_JOB) -> yolo($YOLO_JOB) -> finalize_yolo($FINALIZE_YOLO_JOB) -> p0($P0_JOB) -> b2($B2_JOB) -> config($CONFIG_JOB) -> benchmark($BENCHMARK_JOB) -> finalize_b2($B2_FINAL_JOB)"
printf '%s\n' "manifest=$MANIFEST logs=$PROJECT_ROOT/slurm-thesis-*.out"
printf '%s\n' "Monitor: squeue -u \"$USER\"; sacct -j $VALIDATE_JOB,$YOLO_JOB,$FINALIZE_YOLO_JOB,$P0_JOB,$B2_JOB,$CONFIG_JOB,$BENCHMARK_JOB,$B2_FINAL_JOB --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS"
