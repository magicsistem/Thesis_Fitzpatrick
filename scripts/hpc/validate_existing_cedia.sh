#!/usr/bin/env bash
# Read-only, no-GPU gate after a completed heavy preflight.
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd -P)}
DATA_ROOT=${DATA_ROOT:?Export DATA_ROOT}
FOLD=${1:?Usage: validate_existing_cedia.sh FOLD}
YOLO_ROOT=${YOLO_ROOT:-$PROJECT_ROOT/results/benchmark_v1/yolo}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-$DATA_ROOT/manifests/isic2018_task1_train_disjoint.json}
FOLDS_FILE=${FOLDS_FILE:-$DATA_ROOT/manifests/isic2018_task1_train_disjoint_folds_5.json}
DARKNET_BIN=${DARKNET_BIN:-$PROJECT_ROOT/models/yolov3-darknet/source/darknet}
INITIAL_WEIGHTS=${INITIAL_WEIGHTS:-$PROJECT_ROOT/models/yolov3-darknet/checkpoints/darknet53.conv.74}
cd "$PROJECT_ROOT"
scripts/hpc/run_in_container.sh --check-runtime-paths
# Project Python must never run on the login-node interpreter: CEDIA's host
# Python is older than this repository's supported syntax.  This is a
# read-only resource contract check, not bootstrap or dataset preparation.
scripts/hpc/run_in_container.sh --cpu -- python scripts/hpc/bootstrap_resources.py --require-sources --require-checkpoints
scripts/hpc/run_in_container.sh --cpu -- python scripts/benchmark/yolov3.py resume-plan \
  --darknet "$DARKNET_BIN" --data "$YOLO_ROOT/fold-$FOLD/lesion.data" --cfg "$YOLO_ROOT/fold-$FOLD/lesion-yolov3.cfg" \
  --initial-weights "$INITIAL_WEIGHTS" --fold "$FOLD" --output "$YOLO_ROOT/fold-$FOLD/training"
