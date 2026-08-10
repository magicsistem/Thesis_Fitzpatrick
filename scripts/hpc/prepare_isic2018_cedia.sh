#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd -P)
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/data/raw/isic2018_task1}
case "${1:-}" in
    --dry-run) echo "Would prepare official ISIC 2018 at $DATA_ROOT in one container invocation; test masks stay sealed." ;;
    --execute)
        available_bytes=$(df -PB1 "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
        required_bytes=$((40 * 1024 * 1024 * 1024))
        if [[ ! -f "$DATA_ROOT/manifests/isic2018_task1_test.json" ]] && ((available_bytes < required_bytes)); then
            echo "ISIC preparation requires at least 40 GiB free before the first download" >&2; exit 2
        fi
        echo "isic_free_gib=$((available_bytes / 1024 / 1024 / 1024))"
        export DATA_ROOT; exec "$SCRIPT_DIR/run_in_container.sh" -- bash "$SCRIPT_DIR/prepare_isic2018_inside_container.sh" "$DATA_ROOT" ;;
    *) echo "Usage: DATA_ROOT=/persistent/path scripts/hpc/prepare_isic2018_cedia.sh --dry-run|--execute" >&2; exit 2 ;;
esac
