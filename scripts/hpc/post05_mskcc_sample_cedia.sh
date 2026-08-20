#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "post05 MSKCC sampling failed rc=$rc line=$LINENO utc=$(date -u +%FT%TZ)" >&2; exit "$rc"' ERR
: "${PROJECT_ROOT:?Export PROJECT_ROOT}"
: "${SIF_PATH:?Export SIF_PATH}"
: "${MSKCC_METADATA:?Export path to the acquired MSKCC metadata CSV/JSON}"
: "${MSKCC_IMAGES_ROOT:?Export path to the acquired MSKCC images directory}"
PROJECT_ROOT=$(realpath "$PROJECT_ROOT")
SIF_PATH=$(realpath "$SIF_PATH")
MSKCC_METADATA=$(realpath "$MSKCC_METADATA")
MSKCC_IMAGES_ROOT=$(realpath "$MSKCC_IMAGES_ROOT")
OUT=${MSKCC_OUT:-$PROJECT_ROOT/results/mskcc_post05_pilot}
COLUMN_MAP=${MSKCC_COLUMN_MAP:-}
REQUIRE_KNOWN_ACQUISITION_MODE=${REQUIRE_KNOWN_ACQUISITION_MODE:-0}
for path in "$PROJECT_ROOT" "$SIF_PATH" "$MSKCC_METADATA" "$MSKCC_IMAGES_ROOT"; do
  [[ -e "$path" ]] || { echo "Missing prerequisite: $path" >&2; exit 2; }
done
cd "$PROJECT_ROOT"
export SIF_PATH
registry_args=(--metadata "$MSKCC_METADATA" --images-root "$MSKCC_IMAGES_ROOT" --output "$OUT/registry")
[[ -z "$COLUMN_MAP" ]] || registry_args+=(--column-map "$COLUMN_MAP")
scripts/hpc/run_in_container.sh -- python scripts/benchmark/prepare_mskcc_pilot.py build-registry "${registry_args[@]}"
sample_args=(--registry "$OUT/registry/mskcc_registry.json" --output "$OUT/sample")
[[ "$REQUIRE_KNOWN_ACQUISITION_MODE" == 1 ]] && sample_args+=(--require-known-acquisition-mode)
scripts/hpc/run_in_container.sh -- python scripts/benchmark/prepare_mskcc_pilot.py sample "${sample_args[@]}"
echo "MSKCC sample: $OUT/sample/mskcc_sample_100.json"
