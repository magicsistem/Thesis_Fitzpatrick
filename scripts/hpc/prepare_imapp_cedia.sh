#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd -P)
IMAPP_ROOT=${IMAPP_ROOT:-$PROJECT_ROOT/data/external/imapp_v1_1}
case "${1:-}" in
    --dry-run) echo "Would prepare official IMA++ v1.1 at $IMAPP_ROOT in one container invocation after CC BY-NC-ND 4.0 review." ;;
    --execute)
        available_bytes=$(df -PB1 "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
        required_bytes=$((15 * 1024 * 1024 * 1024))
        if [[ ! -f "$IMAPP_ROOT/manifests/imapp_test.json" ]] && ((available_bytes < required_bytes)); then
            echo "IMA++ preparation requires at least 15 GiB free before the first download" >&2; exit 2
        fi
        echo "imapp_free_gib=$((available_bytes / 1024 / 1024 / 1024))"
        export DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/data}; exec "$SCRIPT_DIR/run_in_container.sh" -- bash "$SCRIPT_DIR/prepare_imapp_inside_container.sh" "$IMAPP_ROOT" ;;
    *) echo "Usage: IMAPP_ROOT=/persistent/path scripts/hpc/prepare_imapp_cedia.sh --dry-run|--execute" >&2; exit 2 ;;
esac
