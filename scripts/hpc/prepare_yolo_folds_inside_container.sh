#!/usr/bin/env bash
# Build all five leakage-safe Darknet datasets/configurations in one container process.
set -Eeuo pipefail

DATA_ROOT=${1:?Pass the persistent ISIC data root}
PROJECT_ROOT=${2:?Pass the repository root}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-$DATA_ROOT/manifests/isic2018_task1_train_disjoint.json}
FOLDS_FILE=${FOLDS_FILE:-$DATA_ROOT/manifests/isic2018_task1_train_disjoint_folds_5.json}
YOLO_ROOT=${YOLO_ROOT:-$PROJECT_ROOT/results/benchmark_v1/yolo}
SOURCE_CFG=$PROJECT_ROOT/models/yolov3-darknet/source/cfg/yolov3.cfg

for path in "$DATA_ROOT" "$TRAIN_MANIFEST" "$FOLDS_FILE" "$SOURCE_CFG"; do
    [[ -e "$path" ]] || { echo "Missing YOLO preparation prerequisite: $path" >&2; exit 2; }
done

for fold in 0 1 2 3 4; do
    fold_dir=$YOLO_ROOT/fold-$fold
    python scripts/benchmark/yolov3.py prepare-fold \
        --manifest "$TRAIN_MANIFEST" --data-root "$DATA_ROOT" \
        --folds "$FOLDS_FILE" --fold "$fold" --output "$fold_dir" --seed 20260806
    python scripts/benchmark/yolov3.py configure \
        --source-cfg "$SOURCE_CFG" --output "$fold_dir/lesion-yolov3.cfg"
    mkdir -p "$fold_dir/backup" "$fold_dir/training"
done

echo "Prepared five YOLO folds under $YOLO_ROOT"
