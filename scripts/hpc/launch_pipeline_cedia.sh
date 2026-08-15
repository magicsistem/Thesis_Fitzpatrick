#!/usr/bin/env bash
# Submit a non-blocking, read-only-validated CEDIA resume DAG.  No preflight.
set -Eeuo pipefail

usage() {
    cat >&2 <<'EOF'
Usage:
  launch_pipeline_cedia.sh --resume-existing --skip-preflight
  launch_pipeline_cedia.sh --recover-existing --skip-preflight --folds 0,3[,4] \\
      --previous-yolo-job JOB --blocked-jobs JOB,JOB,JOB,JOB,JOB,JOB

The recovery list is explicit, must match the read-only validator, and cannot
contain completed folds 1 or 2.  Optional YOLO_EXCLUDE_NODES is an sbatch
--exclude list (for example compute-0-1); it is never enabled by default.
EOF
}

MODE=${1:-}; shift || true
[[ ${1:-} == --skip-preflight ]] || { usage; exit 2; }; shift
PREVIOUS_YOLO_JOB= BLOCKED_JOBS=
case "$MODE" in
    --resume-existing) [[ $# == 0 ]] || { usage; exit 2; }; PENDING_FOLDS=0,2,3,4 ;;
    --recover-existing)
        [[ ${1:-} == --folds && -n ${2:-} && ${3:-} == --previous-yolo-job && ${4:-} =~ ^[0-9]+$ && ${5:-} == --blocked-jobs && -n ${6:-} && $# == 6 ]] || { usage; exit 2; }
        PENDING_FOLDS=$2
        PREVIOUS_YOLO_JOB=$4
        BLOCKED_JOBS=$6
        ;;
    *) usage; exit 2 ;;
esac
[[ "$PENDING_FOLDS" =~ ^[0-4](,[0-4])*$ ]] || { echo "--folds must be comma-separated folds 0..4" >&2; exit 2; }
[[ $(tr ',' '\n' <<< "$PENDING_FOLDS" | sort -u | wc -l) -eq $(tr ',' '\n' <<< "$PENDING_FOLDS" | wc -l) ]] || { echo "--folds must not contain duplicates" >&2; exit 2; }
if [[ "$MODE" == --recover-existing ]]; then
    [[ ",$PENDING_FOLDS," != *,1,* && ",$PENDING_FOLDS," != *,2,* ]] || { echo "folds 1 and 2 are completed/reused and may not be submitted by recovery" >&2; exit 2; }
    [[ "$BLOCKED_JOBS" =~ ^[0-9]+(,[0-9]+){5}$ ]] || { echo "--blocked-jobs must contain exactly six job IDs" >&2; exit 2; }
    [[ $(tr ',' '\n' <<< "$BLOCKED_JOBS" | sort -u | wc -l) == 6 ]] || { echo "--blocked-jobs must not contain duplicates" >&2; exit 2; }
fi

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
YOLO_ARRAY="$PENDING_FOLDS%2"
B2_ARRAY=0-74%2
YOLO_EXCLUDE_NODES=${YOLO_EXCLUDE_NODES:-}
[[ -z "$YOLO_EXCLUDE_NODES" || "$YOLO_EXCLUDE_NODES" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*(,[A-Za-z0-9][A-Za-z0-9._-]*)*$ ]] || { echo "YOLO_EXCLUDE_NODES must be a comma-separated Slurm node list" >&2; exit 2; }

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
    [[ ! -e "$marker" ]] || { echo "Existing downstream result requires inspection; refusing duplicate launch: $marker" >&2; exit 2; }
done
command -v squeue >/dev/null 2>&1 || { echo "squeue is required to reject an active duplicate launch" >&2; exit 2; }
if [[ "$MODE" == --recover-existing ]]; then
    if squeue -h -j "$PREVIOUS_YOLO_JOB" -o '%i %T %R' 2>/dev/null | grep -q .; then
        echo "Previous YOLO array $PREVIOUS_YOLO_JOB still has active tasks; inspect: sacct -j $PREVIOUS_YOLO_JOB --format=JobID,State,ExitCode,Elapsed,NodeList" >&2
        exit 2
    fi
    if squeue -h -j "$BLOCKED_JOBS" -o '%i %T %R' 2>/dev/null | grep -q .; then
        echo "Blocked descendants remain: $BLOCKED_JOBS" >&2
        echo "Inspect: squeue -j $BLOCKED_JOBS" >&2
        echo "Proceed only when $PREVIOUS_YOLO_JOB is terminal and all six are PENDING (Dependency) with no started task." >&2
        echo "Then cancel manually: scancel ${BLOCKED_JOBS//,/ }" >&2
        exit 2
    fi
fi
if squeue -h -u "${USER:-$(id -u)}" -n thesis-yolo -o '%A' 2>/dev/null | grep -q .; then
    echo "An active thesis-yolo task exists; inspect it before recovery and do not submit a duplicate" >&2; exit 2
fi
for manifest in .cedia/resume_pipeline_*.json; do
    [[ -e "$manifest" ]] || continue
    while IFS= read -r job; do
        [[ -z "$job" ]] || ! squeue -h -j "$job" -o '%A' 2>/dev/null | grep -q . || { echo "Active resume job $job from $manifest; refusing duplicate launch" >&2; exit 2; }
    done < <(sed -n 's/.*"job_id":"\([0-9][0-9]*\)".*/\1/p' "$manifest")
done

export PROJECT_ROOT DATA_ROOT SIF_PATH TRAIN_MANIFEST DEVELOPMENT_MANIFEST FOLDS_FILE YOLO_ROOT YOLO_FROZEN_ROOT P0_ROOT B2_ROOT ARTIFACT_ROOT BENCHMARK_CONFIG
export DARKNET_GPU_CONFIRMED=YES YOLO_EXPECTED_PENDING="$PENDING_FOLDS"
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
yolo_args=(--array="$YOLO_ARRAY")
[[ -z "$YOLO_EXCLUDE_NODES" ]] || yolo_args+=(--exclude="$YOLO_EXCLUDE_NODES")
YOLO_JOB=$(submit yolo "$VALIDATE_JOB" "${yolo_args[@]}" scripts/hpc/train_yolo_cedia.slurm)
FINALIZE_YOLO_JOB=$(submit finalize_yolo "$YOLO_JOB" --array=0-4%2 scripts/hpc/finalize_yolo_cedia.slurm)
P0_JOB=$(submit p0 "$FINALIZE_YOLO_JOB" scripts/hpc/prepare_p0_oof_cedia.slurm)
B2_JOB=$(submit b2 "$P0_JOB" --array="$B2_ARRAY" scripts/hpc/train_b2_cedia.slurm)
CONFIG_JOB=$(submit config "$B2_JOB" --job-name=thesis-config --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=60G --gres=gpu:a100-sxm4-40gb:1 --time=00:20:00 --output=slurm-thesis-config-%j.out --error=slurm-thesis-config-%j.err --wrap "cd \"$PROJECT_ROOT\" && \"$PROJECT_ROOT/scripts/hpc/run_in_container.sh\" -- python scripts/hpc/configure_benchmark_cedia.py --frozen-yolo-root \"$YOLO_FROZEN_ROOT\"")
BENCHMARK_JOB=$(submit benchmark "$CONFIG_JOB" scripts/hpc/run_benchmark_cedia.slurm)
B2_FINAL_JOB=$(submit finalize_b2 "$BENCHMARK_JOB" scripts/hpc/finalize_b2_cedia.slurm)

MANIFEST="$PROJECT_ROOT/.cedia/resume_pipeline_$(date -u +%Y%m%d_%H%M%S).json"
MANIFEST_TMP=$(mktemp "$PROJECT_ROOT/.cedia/.resume_pipeline.XXXXXX")
cat > "$MANIFEST_TMP" <<EOF
{"schema_version":2,"status":"submitted","created_utc":"$(date -u +%FT%TZ)","git_sha":"$GIT_HEAD","mode":"${MODE#--}","skip_preflight":true,"previous_yolo_job":"$PREVIOUS_YOLO_JOB","blocked_jobs":"$BLOCKED_JOBS","fold_1_reused":true,"fold_2_reused":true,"yolo_folds":"$PENDING_FOLDS","yolo_exclude_nodes":"$YOLO_EXCLUDE_NODES","b2_tasks":75,"paths":{"project_root":"$PROJECT_ROOT","data_root":"$DATA_ROOT","sif_path":"$SIF_PATH","yolo_root":"$YOLO_ROOT"},"jobs":[{"stage":"validation","job_id":"$VALIDATE_JOB","dependency":null},{"stage":"yolo","job_id":"$YOLO_JOB","dependency":"afterok:$VALIDATE_JOB","array":"$YOLO_ARRAY"},{"stage":"finalize_yolo","job_id":"$FINALIZE_YOLO_JOB","dependency":"afterok:$YOLO_JOB","array":"0-4%2"},{"stage":"p0","job_id":"$P0_JOB","dependency":"afterok:$FINALIZE_YOLO_JOB"},{"stage":"b2","job_id":"$B2_JOB","dependency":"afterok:$P0_JOB","array":"$B2_ARRAY"},{"stage":"config","job_id":"$CONFIG_JOB","dependency":"afterok:$B2_JOB"},{"stage":"benchmark","job_id":"$BENCHMARK_JOB","dependency":"afterok:$CONFIG_JOB"},{"stage":"finalize_b2","job_id":"$B2_FINAL_JOB","dependency":"afterok:$BENCHMARK_JOB"}]}
EOF
mv -f "$MANIFEST_TMP" "$MANIFEST"
printf '%s\n' "Recovery DAG submitted from $GIT_HEAD; preflight=NOT_SUBMITTED folds=$PENDING_FOLDS fold_1=REUSED fold_2=REUSED exclude=${YOLO_EXCLUDE_NODES:-none}"
printf '%s\n' "DAG: validation($VALIDATE_JOB) -> yolo($YOLO_JOB) -> finalize_yolo($FINALIZE_YOLO_JOB) -> p0($P0_JOB) -> b2($B2_JOB) -> config($CONFIG_JOB) -> benchmark($BENCHMARK_JOB) -> finalize_b2($B2_FINAL_JOB)"
printf '%s\n' "manifest=$MANIFEST logs=$PROJECT_ROOT/slurm-thesis-*.out"
printf '%s\n' "Monitor: squeue -u \"${USER:-$(id -u)}\"; sacct -j $VALIDATE_JOB,$YOLO_JOB,$FINALIZE_YOLO_JOB,$P0_JOB,$B2_JOB,$CONFIG_JOB,$BENCHMARK_JOB,$B2_FINAL_JOB --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS"
[[ "$MODE" != --recover-existing ]] || printf '%s\n' "Old blocked jobs are not cancelled: inspect squeue -j $BLOCKED_JOBS; cancel manually only after inspection."
