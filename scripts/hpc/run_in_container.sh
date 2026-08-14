#!/usr/bin/env bash
# Run one complete project stage inside the read-only CEDIA SIF.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=${PROJECT_ROOT:-$SCRIPT_DIR/../..}
PROJECT_ROOT=$(cd -- "$PROJECT_ROOT" && pwd -P)
SIF_PATH=${SIF_PATH:-${HOME:?HOME is required}/pytorch_24.01-py3.sif}
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/data}
VENV_PATH=${VENV_PATH:-$PROJECT_ROOT/.cedia/venv}
EXPECTED_SIF_SHA256=b9db68700a47ae0811e8c4758d6effc3221000dac53e5d05ece0a8cafd77e2a3
USE_NV=1
USE_VENV=1
ALLOW_OTHER_SIF=0
CHECK_RUNTIME_PATHS=0

usage() {
    cat <<'EOF'
Usage: scripts/hpc/run_in_container.sh [--cpu] [--no-venv] [--allow-unverified-sif] -- COMMAND [ARG ...]
       scripts/hpc/run_in_container.sh --check-runtime-paths

Environment: SIF_PATH, DATA_ROOT and VENV_PATH may override their safe defaults.
The command is passed as an argument vector; no shell reconstruction is performed.
EOF
}

while (($#)); do
    case "$1" in
        --cpu) USE_NV=0; shift ;;
        --no-venv) USE_VENV=0; shift ;;
        --allow-unverified-sif) ALLOW_OTHER_SIF=1; shift ;;
        --check-runtime-paths) CHECK_RUNTIME_PATHS=1; shift ;;
        --help|-h) usage; exit 0 ;;
        --) shift; break ;;
        *) echo "Unknown wrapper option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
if ((CHECK_RUNTIME_PATHS)); then
    (($# == 0)) || { echo "--check-runtime-paths does not accept a command" >&2; exit 2; }
else
    (($#)) || { echo "A command is required after --" >&2; exit 2; }
fi

directory_details() {
    local label=$1 path=$2
    printf '%s path=%s uid_gid=%s:%s owner=%s:%s mode=%s filesystem=%s\n' "$label" "$path" \
        "$(stat -c '%u' "$path" 2>/dev/null || echo '?')" "$(stat -c '%g' "$path" 2>/dev/null || echo '?')" \
        "$(stat -c '%U' "$path" 2>/dev/null || echo '?')" "$(stat -c '%G' "$path" 2>/dev/null || echo '?')" \
        "$(stat -c '%A' "$path" 2>/dev/null || echo '?')" "$(df -PT "$path" 2>/dev/null | awk 'NR==2 {print $2 ":" $7}' || echo '?')" >&2
}

ensure_runtime_directory() {
    local label=$1 path=$2 probe
    if ! mkdir -p "$path"; then
        echo "Cannot create $label" >&2; directory_details "$label" "$path"; return 1
    fi
    if [[ ! -d "$path" || ! -w "$path" || ! -x "$path" || "$(stat -c '%u' "$path")" != "$(id -u)" ]]; then
        echo "$label is not an accessible directory owned by this user" >&2; directory_details "$label" "$path"; return 1
    fi
    probe=$(mktemp "$path/.write-probe.XXXXXX") || { echo "$label cannot create files" >&2; directory_details "$label" "$path"; return 1; }
    rm -f "$probe"
}

configure_runtime_paths() {
    local job_key tmp_root
    job_key="${SLURM_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}-${SLURM_STEP_ID:-0}"
    if [[ -n "${SLURM_TMPDIR:-}" && -d "$SLURM_TMPDIR" && -w "$SLURM_TMPDIR" && -x "$SLURM_TMPDIR" ]]; then
        tmp_root="$SLURM_TMPDIR/apptainer-$job_key"
        RUNTIME_TMP_SOURCE=slurm
    else
        [[ -z "${SLURM_TMPDIR:-}" ]] || { echo "SLURM_TMPDIR is unavailable; using user fallback" >&2; directory_details SLURM_TMPDIR "$SLURM_TMPDIR"; }
        tmp_root="$PROJECT_ROOT/.cedia/apptainer-tmp/${USER:-$(id -u)}/jobs/$job_key"
        RUNTIME_TMP_SOURCE=fallback
    fi
    APPTAINER_TMPDIR=$tmp_root
    TMPDIR=$tmp_root
    APPTAINER_CACHEDIR="$PROJECT_ROOT/.cedia/apptainer-cache/${USER:-$(id -u)}"
    ensure_runtime_directory APPTAINER_TMPDIR "$APPTAINER_TMPDIR" || return 2
    ensure_runtime_directory TMPDIR "$TMPDIR" || return 2
    ensure_runtime_directory APPTAINER_CACHEDIR "$APPTAINER_CACHEDIR" || return 2
    SINGULARITY_TMPDIR=$APPTAINER_TMPDIR
    SINGULARITY_CACHEDIR=$APPTAINER_CACHEDIR
    export APPTAINER_TMPDIR APPTAINER_CACHEDIR TMPDIR SINGULARITY_TMPDIR SINGULARITY_CACHEDIR RUNTIME_TMP_SOURCE
    printf 'runtime_paths source=%s tmp=%s cache=%s lock_policy=apptainer-managed-no-pipeline-lock\n' \
        "$RUNTIME_TMP_SOURCE" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
}

configure_runtime_paths || exit $?
((CHECK_RUNTIME_PATHS == 0)) || exit 0
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

printf 'container_stage utc=%s host=%s git=%s sif_sha256=%s device=%s apptainer_tmp=%s apptainer_cache=%s tmpdir=%s\n' \
    "$(date -u +%FT%TZ)" "$(hostname)" "$(git -C "$PROJECT_ROOT" rev-parse HEAD)" "$actual_sif_sha" "$device" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$TMPDIR"
exec "$runtime" "${container_args[@]}" \
    --env "PATH=$path_value" \
    --env "VIRTUAL_ENV=$VENV_PATH" \
    --env "PYTHONPATH=$PROJECT_ROOT/src:$PROJECT_ROOT/scripts" \
    --env "THESIS_ADAPTER_PYTHON=$python_value" \
    --env "THESIS_DEVICE=$device" \
    --env "THESIS_DATA_ROOT=$DATA_ROOT" \
    "$SIF_PATH" "$@"
