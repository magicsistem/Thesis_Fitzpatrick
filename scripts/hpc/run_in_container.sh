#!/usr/bin/env bash
# Run one complete project stage inside the read-only CEDIA SIF.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd -P)
SIF_PATH=${SIF_PATH:-${HOME:?HOME is required}/pytorch_24.01-py3.sif}
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/data}
VENV_PATH=${VENV_PATH:-$PROJECT_ROOT/.cedia/venv}
EXPECTED_SIF_SHA256=b9db68700a47ae0811e8c4758d6effc3221000dac53e5d05ece0a8cafd77e2a3
USE_NV=1
USE_VENV=1
ALLOW_OTHER_SIF=0

usage() {
    cat <<'EOF'
Usage: scripts/hpc/run_in_container.sh [--cpu] [--no-venv] [--allow-unverified-sif] -- COMMAND [ARG ...]

Environment: SIF_PATH, DATA_ROOT and VENV_PATH may override their safe defaults.
The command is passed as an argument vector; no shell reconstruction is performed.
EOF
}

while (($#)); do
    case "$1" in
        --cpu) USE_NV=0; shift ;;
        --no-venv) USE_VENV=0; shift ;;
        --allow-unverified-sif) ALLOW_OTHER_SIF=1; shift ;;
        --help|-h) usage; exit 0 ;;
        --) shift; break ;;
        *) echo "Unknown wrapper option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
(($#)) || { echo "A command is required after --" >&2; exit 2; }
[[ -f "$SIF_PATH" ]] || { echo "SIF not found: $SIF_PATH" >&2; exit 2; }

actual_sif_sha=$(sha256sum "$SIF_PATH" | awk '{print $1}')
if [[ "$actual_sif_sha" != "$EXPECTED_SIF_SHA256" && "$ALLOW_OTHER_SIF" -ne 1 ]]; then
    echo "Unexpected SIF SHA-256: $actual_sif_sha" >&2
    echo "Expected: $EXPECTED_SIF_SHA256" >&2
    echo "Audit another image first, then opt in with --allow-unverified-sif." >&2
    exit 2
fi

runtime=$(command -v apptainer || command -v singularity || true)
[[ -n "$runtime" ]] || { echo "Apptainer or Singularity is required" >&2; exit 2; }
mkdir -p "$PROJECT_ROOT/.cedia" "$DATA_ROOT" "$PROJECT_ROOT/results"

# CEDIA lacks squashfuse and expands a SIF while executing it.  Keep that
# expansion on a persistent, auditable filesystem instead of an opaque /tmp.
APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-${SLURM_TMPDIR:-$PROJECT_ROOT/.cedia/apptainer-tmp}}
APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-$PROJECT_ROOT/.cedia/apptainer-cache}
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
[[ -w "$APPTAINER_TMPDIR" && -w "$APPTAINER_CACHEDIR" ]] || { echo "Apptainer temporary/cache directory is not writable" >&2; exit 2; }
export APPTAINER_TMPDIR APPTAINER_CACHEDIR

binds=(--bind "$PROJECT_ROOT:$PROJECT_ROOT")
if [[ "$DATA_ROOT" != "$PROJECT_ROOT" && "$DATA_ROOT" != "$PROJECT_ROOT"/* ]]; then
    mkdir -p "$DATA_ROOT"
    binds+=(--bind "$DATA_ROOT:$DATA_ROOT")
fi
container_args=(exec)
((USE_NV == 0)) || container_args+=(--nv)
container_args+=("${binds[@]}" --pwd "$PROJECT_ROOT")

path_value=$PATH
python_value=python3
if ((USE_VENV)); then
    [[ -x "$VENV_PATH/bin/python" ]] || {
        echo "Persistent environment missing: $VENV_PATH" >&2
        echo "Run scripts/hpc/bootstrap_cedia.sh --execute first." >&2
        exit 2
    }
    path_value="$VENV_PATH/bin:$PATH"
    python_value="$VENV_PATH/bin/python"
fi
device=cpu
((USE_NV == 0)) || device=cuda

printf 'container_stage utc=%s host=%s git=%s sif_sha256=%s device=%s apptainer_tmp=%s apptainer_cache=%s\n' \
    "$(date -u +%FT%TZ)" "$(hostname)" "$(git -C "$PROJECT_ROOT" rev-parse HEAD)" "$actual_sif_sha" "$device" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
exec "$runtime" "${container_args[@]}" \
    --env "PATH=$path_value" \
    --env "VIRTUAL_ENV=$VENV_PATH" \
    --env "PYTHONPATH=$PROJECT_ROOT/src:$PROJECT_ROOT/scripts" \
    --env "THESIS_ADAPTER_PYTHON=$python_value" \
    --env "THESIS_DEVICE=$device" \
    --env "THESIS_DATA_ROOT=$DATA_ROOT" \
    "$SIF_PATH" "$@"
