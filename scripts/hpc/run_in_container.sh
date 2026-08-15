#!/usr/bin/env bash
# Run one complete project stage inside the read-only CEDIA SIF.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=${PROJECT_ROOT:-$SCRIPT_DIR/../..}
PROJECT_ROOT=$(cd -- "$PROJECT_ROOT" && pwd -P)
SIF_PATH=${SIF_PATH:-${HOME:?HOME is required}/pytorch_24.01-py3.sif}
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/data}
VENV_PATH=${VENV_PATH:-$PROJECT_ROOT/.cedia/venv}
USE_NV=1
USE_VENV=1
CHECK_RUNTIME_PATHS=0

usage() {
    cat <<'EOF'
Usage: scripts/hpc/run_in_container.sh [--cpu] [--no-venv] -- COMMAND [ARG ...]
       scripts/hpc/run_in_container.sh --check-runtime-paths

Environment: SIF_PATH, DATA_ROOT and VENV_PATH may override their safe defaults.
The command is passed as an argument vector; no shell reconstruction is performed.
EOF
}

while (($#)); do
    case "$1" in
        --cpu) USE_NV=0; shift ;;
        --no-venv) USE_VENV=0; shift ;;
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

# Nested project stages are already in the SIF.  Do not recursively invoke a
# container runtime or recreate temporary sandboxes.
if [[ "${THESIS_IN_CONTAINER:-}" == 1 ]]; then
    ((CHECK_RUNTIME_PATHS == 0)) || exit 0
    printf 'container_stage nested=1 runtime=none action=direct-exec\n'
    exec "$@"
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
    ensure_runtime_directory APPTAINER_TMPDIR_BASE "$tmp_root" || return 2
    APPTAINER_TMPDIR=$(mktemp -d "$tmp_root/invocation.XXXXXX") || {
        echo "Cannot create an isolated Apptainer invocation directory" >&2; directory_details APPTAINER_TMPDIR_BASE "$tmp_root"; return 2
    }
    RUNTIME_TMP_BASE=$tmp_root
    TMPDIR=$APPTAINER_TMPDIR
    APPTAINER_CACHEDIR="$PROJECT_ROOT/.cedia/apptainer-cache/${USER:-$(id -u)}"
    ensure_runtime_directory APPTAINER_TMPDIR "$APPTAINER_TMPDIR" || return 2
    ensure_runtime_directory TMPDIR "$TMPDIR" || return 2
    ensure_runtime_directory APPTAINER_CACHEDIR "$APPTAINER_CACHEDIR" || return 2
    SINGULARITY_TMPDIR=$APPTAINER_TMPDIR
    SINGULARITY_CACHEDIR=$APPTAINER_CACHEDIR
    export APPTAINER_TMPDIR APPTAINER_CACHEDIR TMPDIR SINGULARITY_TMPDIR SINGULARITY_CACHEDIR RUNTIME_TMP_SOURCE RUNTIME_TMP_BASE
    printf 'runtime_paths source=%s tmp=%s cache=%s lock_policy=apptainer-managed-no-pipeline-lock\n' \
        "$RUNTIME_TMP_SOURCE" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
}

configure_runtime_paths || exit $?
cleanup_runtime() {
    local rc=$?
    trap - EXIT
    if [[ -n "${RUNTIME_DIAGNOSTICS_STARTED:-}" ]]; then runtime_diag_stop || true; fi
    if [[ -n "${RUNTIME_TMP_BASE:-}" && "$APPTAINER_TMPDIR" == "$RUNTIME_TMP_BASE"/invocation.* && -d "$APPTAINER_TMPDIR" ]]; then
        rm -rf -- "$APPTAINER_TMPDIR" || echo "Could not remove isolated Apptainer temporary directory: $APPTAINER_TMPDIR" >&2
    fi
    exit "$rc"
}
trap cleanup_runtime EXIT
((CHECK_RUNTIME_PATHS == 0)) || exit 0
[[ -f "$SIF_PATH" ]] || { echo "SIF not found: $SIF_PATH" >&2; exit 2; }

if runtime=$(command -v singularity 2>/dev/null); then
    runtime_kind=singularity
elif runtime=$(command -v apptainer 2>/dev/null); then
    runtime_kind=apptainer-fallback
else
    echo "Neither singularity nor apptainer is available on this Slurm node; no checkpoint was touched." >&2
    exit 2
fi
runtime_version=$($runtime --version 2>&1 | head -n 1 || true)
sif_metadata=$(stat -c 'bytes=%s mtime_epoch=%Y owner=%u:%g' "$SIF_PATH")
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
container_host_started_epoch=$(date +%s)

printf 'container_stage utc=%s host=%s git=%s sif_identity=%s device=%s apptainer_tmp=%s apptainer_cache=%s tmpdir=%s\n' \
    "$(date -u +%FT%TZ)" "$(hostname)" "$(git -C "$PROJECT_ROOT" rev-parse HEAD)" "not-computed:$sif_metadata" "$device" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$TMPDIR"
printf 'container_runtime_choice path=%s kind=%s version=%s sif=%s %s\n' "$runtime" "$runtime_kind" "$runtime_version" "$SIF_PATH" "$sif_metadata"
if [[ -n "${RUNTIME_DIAGNOSTICS_LOG:-}" ]]; then
    mkdir -p "$(dirname -- "$RUNTIME_DIAGNOSTICS_LOG")"
    printf 'container_runtime utc=%s job=%s task=%s runtime=%s kind=%s version=%s sif=%s tmp=%s cache=%s tmpdir=%s\n' \
        "$(date -u +%FT%TZ)" "${SLURM_JOB_ID:-none}" "${SLURM_ARRAY_TASK_ID:-none}" "$runtime" "$runtime_kind" "$runtime_version" "$SIF_PATH" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$TMPDIR" >> "$RUNTIME_DIAGNOSTICS_LOG"
fi
"$runtime" "${container_args[@]}" \
    --env "PATH=$path_value" \
    --env "VIRTUAL_ENV=$VENV_PATH" \
    --env "PYTHONPATH=$PROJECT_ROOT/src:$PROJECT_ROOT/scripts" \
    --env "THESIS_ADAPTER_PYTHON=$python_value" \
    --env "THESIS_DEVICE=$device" \
    --env "THESIS_DATA_ROOT=$DATA_ROOT" \
    --env "THESIS_CONTAINER_HOST_STARTED_EPOCH=$container_host_started_epoch" \
    --env "THESIS_IN_CONTAINER=1" \
    --env "PYTHONNOUSERSITE=1" \
    "$SIF_PATH" "$@" &
payload_pid=$!
if [[ -n "${RUNTIME_DIAGNOSTICS_LOG:-}" ]]; then
    source "$SCRIPT_DIR/runtime_diagnostics.sh"
    runtime_diag_start "$RUNTIME_DIAGNOSTICS_LOG" 60 "$payload_pid"
    RUNTIME_DIAGNOSTICS_STARTED=1
fi
set +e
wait "$payload_pid"
payload_rc=$?
set -e
exit "$payload_rc"
