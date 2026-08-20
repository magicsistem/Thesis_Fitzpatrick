#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "post05 preflight failed rc=$rc line=$LINENO utc=$(date -u +%FT%TZ)" >&2; exit "$rc"' ERR
: "${PROJECT_ROOT:?Export PROJECT_ROOT}"
: "${SIF_PATH:?Export SIF_PATH}"
PROJECT_ROOT=$(realpath "$PROJECT_ROOT")
SIF_PATH=$(realpath "$SIF_PATH")
CV_YOLO_ROOT=${CV_YOLO_ROOT:-$PROJECT_ROOT/results/benchmark_v1/yolo}
cd "$PROJECT_ROOT"
[[ -z "$(git status --porcelain)" ]] || { echo "Git tree is dirty; commit/push/pull before scientific execution" >&2; git status --short; exit 2; }
for path in "$SIF_PATH" \
  src/thesis_fitzpatrick/post05.py \
  scripts/benchmark/yolo_final_refit.py \
  scripts/benchmark/run_component_ablation.py \
  scripts/benchmark/select_post05_models.py \
  scripts/benchmark/prepare_mskcc_pilot.py \
  scripts/benchmark/post05_selfcheck.py; do
  [[ -e "$path" ]] || { echo "Missing prerequisite: $path" >&2; exit 2; }
done
export SIF_PATH
scripts/hpc/run_in_container.sh -- python -m py_compile \
  src/thesis_fitzpatrick/post05.py \
  scripts/benchmark/yolo_final_refit.py \
  scripts/benchmark/run_component_ablation.py \
  scripts/benchmark/select_post05_models.py \
  scripts/benchmark/prepare_mskcc_pilot.py \
  scripts/benchmark/post05_selfcheck.py
scripts/hpc/run_in_container.sh -- python scripts/benchmark/post05_selfcheck.py
# Run pytest too when the container already exposes it; the scientific pipeline
# does not install extra packages merely to satisfy this optional check.
if scripts/hpc/run_in_container.sh -- python -c 'import pytest' >/dev/null 2>&1; then
  scripts/hpc/run_in_container.sh -- python -m pytest -q tests/test_post05.py
fi
if [[ -d "$CV_YOLO_ROOT/fold-0" ]]; then
  scripts/hpc/run_in_container.sh -- python scripts/benchmark/yolov3.py validate-folds --root "$CV_YOLO_ROOT"
fi
echo "post05 preflight OK git=$(git rev-parse HEAD) utc=$(date -u +%FT%TZ)"
