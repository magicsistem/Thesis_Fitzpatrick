#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "post05 freeze final YOLO failed rc=$rc line=$LINENO utc=$(date -u +%FT%TZ)" >&2; exit "$rc"' ERR
: "${PROJECT_ROOT:?Export PROJECT_ROOT}"
: "${SIF_PATH:?Export SIF_PATH}"
PROJECT_ROOT=$(realpath "$PROJECT_ROOT")
SIF_PATH=$(realpath "$SIF_PATH")
CV_YOLO_ROOT=${CV_YOLO_ROOT:-$PROJECT_ROOT/results/benchmark_v1/yolo}
FINAL_YOLO_ROOT=${FINAL_YOLO_ROOT:-$CV_YOLO_ROOT/final}
BASE_CONFIG=${BASE_CONFIG:-$PROJECT_ROOT/.cedia/benchmark.cedia.json}
FINAL_B1_CONFIG=${FINAL_B1_CONFIG:-$PROJECT_ROOT/.cedia/post05/benchmark.final_b1.json}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-$PROJECT_ROOT/results/benchmark_v1}
mkdir -p "$(dirname "$FINAL_B1_CONFIG")"
for path in "$PROJECT_ROOT" "$SIF_PATH" "$FINAL_YOLO_ROOT/training/training_state.json" "$BASE_CONFIG"; do
  [[ -e "$path" ]] || { echo "Missing prerequisite: $path" >&2; exit 2; }
done
cd "$PROJECT_ROOT"
export SIF_PATH
scripts/hpc/run_in_container.sh -- python scripts/benchmark/yolo_final_refit.py freeze \
  --root "$FINAL_YOLO_ROOT" --cv-frozen-root "$CV_YOLO_ROOT" --output "$FINAL_YOLO_ROOT/frozen.json"
scripts/hpc/run_in_container.sh -- python scripts/benchmark/yolo_final_refit.py configure-b1 \
  --frozen "$FINAL_YOLO_ROOT/frozen.json" --base-config "$BASE_CONFIG" \
  --output "$FINAL_B1_CONFIG" --artifact-root "$ARTIFACT_ROOT"
echo "Final YOLO frozen: $FINAL_YOLO_ROOT/frozen.json"
echo "Final B1 config:    $FINAL_B1_CONFIG"
