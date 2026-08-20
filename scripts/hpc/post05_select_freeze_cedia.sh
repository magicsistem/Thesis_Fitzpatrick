#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "post05 select/freeze failed rc=$rc line=$LINENO utc=$(date -u +%FT%TZ)" >&2; exit "$rc"' ERR
: "${PROJECT_ROOT:?Export PROJECT_ROOT}"
: "${SIF_PATH:?Export SIF_PATH}"
: "${B1_RUN_ID:?Export completed final-B1 RUN_ID}"
: "${ABLATION_RUN_PREFIX:?Export the common prefix used by post05_run_ablations_cedia.slurm}"
PROJECT_ROOT=$(realpath "$PROJECT_ROOT")
SIF_PATH=$(realpath "$SIF_PATH")
ARTIFACT_ROOT=${ARTIFACT_ROOT:-$PROJECT_ROOT/results/benchmark_v1}
FINAL_YOLO=${FINAL_YOLO:-$ARTIFACT_ROOT/yolo/final/frozen.json}
SELECTION_ROOT=${SELECTION_ROOT:-$ARTIFACT_ROOT/post05_selection}
FREEZE_PATH=${FREEZE_PATH:-$ARTIFACT_ROOT/post05_scientific_freeze.json}
B1_RUN=$ARTIFACT_ROOT/runs/$B1_RUN_ID
conditions=(b1_full b1_no_hair b1_no_yolo b1_no_fov b1_no_post)
args=()
for condition in "${conditions[@]}"; do
  run="$ARTIFACT_ROOT/runs/${ABLATION_RUN_PREFIX}-${condition}"
  [[ -f "$run/run_manifest.json" ]] || { echo "Missing ablation run: $run" >&2; exit 2; }
  args+=(--ablation-run "$run")
done
for path in "$B1_RUN/run_manifest.json" "$FINAL_YOLO" "$SIF_PATH"; do
  [[ -e "$path" ]] || { echo "Missing prerequisite: $path" >&2; exit 2; }
done
cd "$PROJECT_ROOT"
export SIF_PATH
scripts/hpc/run_in_container.sh -- python scripts/benchmark/select_post05_models.py select \
  --b1-run "$B1_RUN" "${args[@]}" --output "$SELECTION_ROOT"
scripts/hpc/run_in_container.sh -- python scripts/benchmark/select_post05_models.py freeze \
  --final-yolo "$FINAL_YOLO" --selection "$SELECTION_ROOT" --output "$FREEZE_PATH"
echo "Selection: $SELECTION_ROOT/segmenter_selection.json"
echo "Freeze:    $FREEZE_PATH"
