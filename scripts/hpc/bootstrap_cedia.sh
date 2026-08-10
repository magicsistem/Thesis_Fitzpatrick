#!/usr/bin/env bash
# Idempotent entry point for a clean CEDIA clone. It never modifies the SIF.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd -P)
MODE=models
EXECUTE=0

usage() {
    cat <<'EOF'
Usage: scripts/hpc/bootstrap_cedia.sh [--dependencies-only] (--dry-run | --execute)

--dry-run prints the validated plan without downloads or installations.
--execute creates .cedia/venv, clones pinned sources, downloads verified public
checkpoints, and builds the pinned Darknet source for the visible A100 toolchain.
Datasets are intentionally prepared by manage_datasets.py after license review.
EOF
}
while (($#)); do
    case "$1" in
        --dependencies-only) MODE=dependencies; shift ;;
        --dry-run) EXECUTE=0; shift ;;
        --execute) EXECUTE=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

SIF_PATH=${SIF_PATH:-${HOME:?HOME is required}/pytorch_24.01-py3.sif}
expected_sha=b9db68700a47ae0811e8c4758d6effc3221000dac53e5d05ece0a8cafd77e2a3
echo "project=$PROJECT_ROOT"
echo "sif=$SIF_PATH expected_sha256=$expected_sha present=$([[ -f "$SIF_PATH" ]] && echo yes || echo no)"
echo "mode=$MODE persistent_venv=$PROJECT_ROOT/.cedia/venv"
echo "One container invocation will perform the complete stage."
if ((EXECUTE == 0)); then
    python3 "$SCRIPT_DIR/bootstrap_resources.py" --clone-sources
    echo "DRY RUN: no directories, packages, downloads, sources, or weights changed."
    exit 0
fi

[[ -f "$SIF_PATH" ]] || { echo "SIF missing: $SIF_PATH" >&2; exit 2; }
actual_sha=$(sha256sum "$SIF_PATH" | awk '{print $1}')
[[ "$actual_sha" == "$expected_sha" ]] || { echo "SIF SHA-256 mismatch: $actual_sha" >&2; exit 2; }
command -v apptainer >/dev/null 2>&1 || command -v singularity >/dev/null 2>&1 || { echo "Apptainer/Singularity not found" >&2; exit 2; }
command -v git >/dev/null 2>&1 || { echo "git not found" >&2; exit 2; }
available_bytes=$(df -PB1 "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
required_bytes=$((5 * 1024 * 1024 * 1024))
((available_bytes >= required_bytes)) || { echo "At least 5 GiB free is required for dependencies, sources and checkpoint archives" >&2; exit 2; }
echo "bootstrap_free_gib=$((available_bytes / 1024 / 1024 / 1024)) required_gib=5"

# Source checkouts do not require the container and remain resumable/persistent.
python3 "$SCRIPT_DIR/bootstrap_resources.py" --clone-sources --execute

exec "$SCRIPT_DIR/run_in_container.sh" --no-venv -- \
    bash "$SCRIPT_DIR/bootstrap_inside_container.sh" "$MODE"
