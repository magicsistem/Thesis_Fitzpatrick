#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "post05 prepare final YOLO failed rc=$rc line=$LINENO utc=$(date -u +%FT%TZ)" >&2; exit "$rc"' ERR

: "${PROJECT_ROOT:?Export PROJECT_ROOT}"
: "${DATA_ROOT:?Export DATA_ROOT}"
: "${SIF_PATH:?Export SIF_PATH}"
PROJECT_ROOT=$(realpath "$PROJECT_ROOT")
DATA_ROOT=$(realpath "$DATA_ROOT")
SIF_PATH=$(realpath "$SIF_PATH")
TRAIN_MANIFEST=${TRAIN_MANIFEST:-$DATA_ROOT/manifests/isic2018_task1_train_disjoint.json}
FOLDS_FILE=${FOLDS_FILE:-$DATA_ROOT/manifests/isic2018_task1_train_disjoint_folds_5.json}
SOURCE_CFG=${SOURCE_CFG:-$PROJECT_ROOT/models/yolov3-darknet/source/cfg/yolov3.cfg}
FINAL_YOLO_ROOT=${FINAL_YOLO_ROOT:-$PROJECT_ROOT/results/benchmark_v1/yolo/final}

for path in "$PROJECT_ROOT" "$DATA_ROOT" "$SIF_PATH" "$TRAIN_MANIFEST" "$FOLDS_FILE" "$SOURCE_CFG"; do
  [[ -e "$path" ]] || { echo "Missing prerequisite: $path" >&2; exit 2; }
done
[[ ! -e "$FINAL_YOLO_ROOT/frozen.json" ]] || { echo "Final YOLO is already frozen; refusing to prepare again" >&2; exit 2; }
cd "$PROJECT_ROOT"
export DATA_ROOT SIF_PATH
exec scripts/hpc/run_in_container.sh -- python scripts/benchmark/yolo_final_refit.py prepare \
  --manifest "$TRAIN_MANIFEST" --data-root "$DATA_ROOT" --folds "$FOLDS_FILE" \
  --source-cfg "$SOURCE_CFG" --output "$FINAL_YOLO_ROOT"
