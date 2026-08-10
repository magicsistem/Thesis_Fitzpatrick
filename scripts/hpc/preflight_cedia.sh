#!/usr/bin/env bash
# Validate host scheduling commands and all container requirements in one SIF execution.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd -P)
SCOPE=code
MANIFEST=
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/data}

usage() { echo "Usage: scripts/hpc/preflight_cedia.sh --scope code|models|smoke|yolo|b2|benchmark [--manifest FILE] [--data-root DIR]"; }
while (($#)); do
    case "$1" in
        --scope) SCOPE=${2:?}; shift 2 ;;
        --manifest) MANIFEST=${2:?}; shift 2 ;;
        --data-root) DATA_ROOT=${2:?}; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
case "$SCOPE" in code|models|smoke|yolo|b2|benchmark) ;; *) echo "Invalid scope: $SCOPE" >&2; exit 2;; esac

host_errors=0
for command_name in sbatch squeue sacct sinfo; do
    if command -v "$command_name" >/dev/null 2>&1; then
        echo "host_command $command_name=$(command -v "$command_name")"
    else
        echo "ERROR host command missing: $command_name" >&2
        host_errors=$((host_errors + 1))
    fi
done
[[ -w "$PROJECT_ROOT" ]] || { echo "ERROR project is not writable: $PROJECT_ROOT" >&2; host_errors=$((host_errors + 1)); }
[[ -d "$DATA_ROOT" ]] || { echo "ERROR data root is missing: $DATA_ROOT" >&2; host_errors=$((host_errors + 1)); }
((host_errors == 0)) || exit 2

args=(python scripts/hpc/preflight_inside_container.py --scope "$SCOPE" --data-root "$DATA_ROOT")
[[ -z "$MANIFEST" ]] || args+=(--manifest "$MANIFEST")
export DATA_ROOT
exec "$SCRIPT_DIR/run_in_container.sh" -- "${args[@]}"
