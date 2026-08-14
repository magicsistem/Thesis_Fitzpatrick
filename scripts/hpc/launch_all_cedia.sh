#!/usr/bin/env bash
# Submit the complete leakage-safe CEDIA benchmark dependency graph.
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
TRAIN_MANIFEST="$MANIFEST_ROOT/isic2018_task1_train_disjoint.json"
DEVELOPMENT_MANIFEST="$MANIFEST_ROOT/isic2018_task1_validation_disjoint.json"
FOLDS_FILE="$MANIFEST_ROOT/isic2018_task1_train_disjoint_folds_5.json"
YOLO_FROZEN_ROOT=${YOLO_FROZEN_ROOT:-$PROJECT_ROOT/results/benchmark_v1/yolo}
BENCHMARK_CONFIG=${BENCHMARK_CONFIG:-$PROJECT_ROOT/.cedia/benchmark.cedia.json}
PIPELINE_BRANCH=${PIPELINE_BRANCH:-fix/hpc-pipeline-validation}
JOB_FILE="$PROJECT_ROOT/.cedia/cedia_job_chain_$(date +%Y%m%d_%H%M%S).txt"
CURRENT_STAGE=initialization

fail() {
    rc=$?
    echo "Pipeline launcher failed rc=$rc stage=$CURRENT_STAGE" >&2
    [[ ! -f "$JOB_FILE" ]] || { echo "Submission registry at failure (job IDs appear only if sbatch succeeded):" >&2; cat "$JOB_FILE" >&2; }
    exit "$rc"
}
trap fail ERR
cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/.cedia"

CURRENT_STAGE="Git traceability check"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "$PIPELINE_BRANCH" ]] || { echo "Expected branch $PIPELINE_BRANCH" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || { echo "Git worktree must be clean before submission" >&2; git status --short >&2; exit 2; }
GIT_HEAD=$(git rev-parse HEAD)
git merge-base --is-ancestor "$GIT_HEAD" "origin/$PIPELINE_BRANCH" || { echo "HEAD is not published on origin/$PIPELINE_BRANCH" >&2; exit 2; }

export PROJECT_ROOT DATA_ROOT SIF_PATH TRAIN_MANIFEST DEVELOPMENT_MANIFEST FOLDS_FILE YOLO_FROZEN_ROOT BENCHMARK_CONFIG

CURRENT_STAGE="bootstrap"
bash scripts/hpc/bootstrap_cedia.sh --execute
python3 scripts/hpc/bootstrap_resources.py --require-sources --require-checkpoints

CURRENT_STAGE="ISIC 2018 disjoint preparation"
bash scripts/hpc/prepare_isic2018_cedia.sh --execute

CURRENT_STAGE="disjoint artifact gate"
python3 - "$TRAIN_MANIFEST" "$DEVELOPMENT_MANIFEST" "$DATA_ROOT/manifests/isic2018_task1_test.json" "$FOLDS_FILE" "$DATA_ROOT/manifests/isic2018_task1_disjoint_overlap_audit.json" <<'PY'
import json, pathlib, sys
train_path, validation_path, test_path, folds_path, audit_path = map(pathlib.Path, sys.argv[1:])
documents = [json.loads(path.read_text(encoding="utf-8")) for path in (train_path, validation_path, test_path)]
counts = [len(document["items"]) for document in documents]
expected = [2522, 64, 1000]
if counts != expected: raise SystemExit(f"Disjoint count gate failed: {counts} != {expected}")
if [document.get("expected_count") for document in documents[:2]] != expected[:2]:
    raise SystemExit("Derived manifest metadata does not match real counts")
audit = json.loads(audit_path.read_text(encoding="utf-8"))
if not audit.get("passed") or audit.get("collisions"): raise SystemExit("Cross-split overlap audit failed")
folds = json.loads(folds_path.read_text(encoding="utf-8"))
if folds.get("seed") != 20260806 or folds.get("fold_count") != 5: raise SystemExit("Unexpected fold contract")
if sum(len(fold["validation_ids"]) for fold in folds["folds"]) != 2522: raise SystemExit("Folds do not cover disjoint train exactly once")
print("Disjoint gate passed: train=2522 validation=64 test=1000 folds=5 audit=0 collisions")
PY

CURRENT_STAGE="tests and preflight"
bash scripts/hpc/preflight_cedia.sh --scope smoke --manifest "$TRAIN_MANIFEST" --data-root "$DATA_ROOT"
scripts/hpc/run_in_container.sh -- python -m unittest discover -s tests -v
bash scripts/hpc/preflight_cedia.sh --scope yolo --manifest "$TRAIN_MANIFEST" --data-root "$DATA_ROOT"

CURRENT_STAGE="YOLO fold preparation"
export DARKNET_GPU_CONFIRMED=YES
bash scripts/hpc/prepare_yolo_folds_cedia.sh --execute

submit() {
    local variable=$1
    shift
    local raw job
    raw=$(sbatch --parsable "$@")
    job=${raw%%;*}
    printf '%s=%s\n' "$variable" "$job" | tee -a "$JOB_FILE" >&2
    printf '%s' "$job"
}

CURRENT_STAGE="Slurm submission"
printf 'GIT_HEAD=%s\nPIPELINE_BRANCH=%s\n' "$GIT_HEAD" "$PIPELINE_BRANCH" | tee "$JOB_FILE"
export SMOKE_MANIFEST="$TRAIN_MANIFEST"
SMOKE_JOB=$(submit SMOKE_JOB scripts/hpc/smoke_cedia.slurm)
YOLO_JOB=$(submit YOLO_JOB --dependency="afterok:$SMOKE_JOB" scripts/hpc/train_yolo_cedia.slurm)
FINALIZE_YOLO_JOB=$(submit FINALIZE_YOLO_JOB --dependency="afterok:$YOLO_JOB" scripts/hpc/finalize_yolo_cedia.slurm)
P0_JOB=$(submit P0_JOB --dependency="afterok:$FINALIZE_YOLO_JOB" scripts/hpc/prepare_p0_oof_cedia.slurm)
B2_JOB=$(submit B2_JOB --dependency="afterok:$P0_JOB" scripts/hpc/train_b2_cedia.slurm)
CONFIG_JOB=$(submit CONFIG_JOB --dependency="afterok:$FINALIZE_YOLO_JOB" --job-name=thesis-config --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=60G --gres=gpu:a100-sxm4-40gb:1 --time=00:20:00 --output=slurm-thesis-config-%j.out --error=slurm-thesis-config-%j.err --wrap='cd "$PROJECT_ROOT" && "$PROJECT_ROOT/scripts/hpc/run_in_container.sh" -- python scripts/hpc/configure_benchmark_cedia.py --frozen-yolo-root "$YOLO_FROZEN_ROOT"')
BENCHMARK_JOB=$(submit BENCHMARK_JOB --dependency="afterok:$CONFIG_JOB" scripts/hpc/run_benchmark_cedia.slurm)
B2_FINAL_JOB=$(submit B2_FINAL_JOB --dependency="afterok:$B2_JOB:$CONFIG_JOB" scripts/hpc/finalize_b2_cedia.slurm)

trap - ERR
echo "Submitted leakage-safe pipeline from $GIT_HEAD"
echo "Job registry: $JOB_FILE"
squeue -u "$USER"
exit 0
