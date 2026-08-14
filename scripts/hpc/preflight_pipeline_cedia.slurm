#!/usr/bin/env bash
# Heavy, idempotent preparation belongs in a batch job, never in the launcher.
#SBATCH --job-name=thesis-preflight
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail
: "${PROJECT_ROOT:?}"; : "${DATA_ROOT:?}"; : "${SIF_PATH:?}"
PROJECT_ROOT=$(realpath "$PROJECT_ROOT"); DATA_ROOT=$(realpath "$DATA_ROOT")
cd "$PROJECT_ROOT"
export PROJECT_ROOT DATA_ROOT SIF_PATH
mkdir -p .cedia
scripts/hpc/run_in_container.sh --check-runtime-paths
source scripts/hpc/runtime_diagnostics.sh
runtime_diag_start "$PROJECT_ROOT/.cedia/diagnostics/preflight-${SLURM_JOB_ID}.log"
cleanup() { runtime_diag_stop; }
trap cleanup EXIT
trap 'runtime_diag_note_signal TERM; exit 143' TERM INT USR1

bash scripts/hpc/bootstrap_cedia.sh --execute
scripts/hpc/run_in_container.sh --cpu -- python scripts/hpc/bootstrap_resources.py --require-sources --require-checkpoints
bash scripts/hpc/prepare_isic2018_cedia.sh --execute
scripts/hpc/run_in_container.sh --cpu -- python - "$DATA_ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]) / "manifests"
names = ("isic2018_task1_train_disjoint.json", "isic2018_task1_validation_disjoint.json", "isic2018_task1_test.json")
documents = [json.loads((root / name).read_text(encoding="utf-8")) for name in names]
if [len(item["items"]) for item in documents] != [2522, 64, 1000]: raise SystemExit("disjoint count gate failed")
audit = json.loads((root / "isic2018_task1_disjoint_overlap_audit.json").read_text(encoding="utf-8"))
folds = json.loads((root / "isic2018_task1_train_disjoint_folds_5.json").read_text(encoding="utf-8"))
if not audit.get("passed") or audit.get("collisions") or folds.get("fold_count") != 5: raise SystemExit("disjoint contract failed")
print("disjoint gate passed")
PY
TRAIN_MANIFEST=${TRAIN_MANIFEST:-$DATA_ROOT/manifests/isic2018_task1_train_disjoint.json}
bash scripts/hpc/preflight_cedia.sh --scope smoke --manifest "$TRAIN_MANIFEST" --data-root "$DATA_ROOT"
scripts/hpc/run_in_container.sh -- python -m unittest discover -s tests -v
bash scripts/hpc/preflight_cedia.sh --scope yolo --manifest "$TRAIN_MANIFEST" --data-root "$DATA_ROOT"
export DARKNET_GPU_CONFIRMED=YES
bash scripts/hpc/prepare_yolo_folds_cedia.sh --execute
