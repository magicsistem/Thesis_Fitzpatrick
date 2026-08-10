#!/usr/bin/env bash
# Prepare all YOLO folds with one SIF conversion; no training is submitted here.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd -P)
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/data/raw/isic2018_task1}
MODE=${1:---dry-run}

case "$MODE" in
    --dry-run)
        echo "project=$PROJECT_ROOT"
        echo "data_root=$DATA_ROOT"
        echo "train_manifest=${TRAIN_MANIFEST:-$DATA_ROOT/manifests/isic2018_task1_train_disjoint.json}"
        echo "folds=${FOLDS_FILE:-$DATA_ROOT/manifests/isic2018_task1_train_disjoint_folds_5.json}"
        echo "DRY RUN: execution would use one CPU container invocation."
        ;;
    --execute)
        export DATA_ROOT
        exec "$SCRIPT_DIR/run_in_container.sh" --cpu -- \
            bash "$SCRIPT_DIR/prepare_yolo_folds_inside_container.sh" "$DATA_ROOT" "$PROJECT_ROOT"
        ;;
    --help|-h)
        echo "Usage: scripts/hpc/prepare_yolo_folds_cedia.sh [--dry-run|--execute]"
        ;;
    *)
        echo "Unknown option: $MODE" >&2
        exit 2
        ;;
esac
