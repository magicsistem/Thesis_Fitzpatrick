#!/usr/bin/env bash
# Source this file from a Slurm script.  It writes directly to one log file.

runtime_diag_snapshot() {
    local label=${1:?label required} target=${2:-${RUNTIME_DIAGNOSTICS_TARGET_PID:-}}
    {
        printf 'diagnostic utc=%s label=%s job=%s task=%s host=%s pid=%s pgid=%s target_pid=%s target_pgid=%s\n' \
            "$(date -u +%FT%TZ)" "$label" "${SLURM_JOB_ID:-none}" "${SLURM_ARRAY_TASK_ID:-none}" "$(hostname)" "$$" "$(ps -o pgid= -p $$ | tr -d ' ')" "$target" "$(ps -o pgid= -p "$target" 2>/dev/null | tr -d ' ' || true)"
        printf 'tmp TMPDIR=%s APPTAINER_TMPDIR=%s APPTAINER_CACHEDIR=%s SLURM_TMPDIR=%s\n' \
            "${TMPDIR:-}" "${APPTAINER_TMPDIR:-}" "${APPTAINER_CACHEDIR:-}" "${SLURM_TMPDIR:-}"
        printf 'processes\n'; ps -eo pid=,ppid=,pgid=,sid=,stat=,etime=,rss=,args= | awk -v sid="$(ps -o sid= -p $$ | tr -d ' ')" '$4 == sid || NR == 1' || true
        printf 'darknet_processes\n'; ps -eo pid=,ppid=,pgid=,sid=,stat=,etime=,rss=,args= | awk '$0 ~ /(^|[[:space:]])[^[:space:]]*darknet([[:space:]]|$)/' || true
        printf 'cgroup\n'; cat /proc/self/cgroup 2>&1 || true
        while IFS= read -r item; do
            [[ -r "$item" ]] && { printf '%s=' "$item"; cat "$item"; }
        done < <(find /sys/fs/cgroup -maxdepth 3 -type f \( -name memory.current -o -name memory.events -o -name memory.max -o -name pids.current -o -name pids.events \) 2>/dev/null | head -40)
        printf 'memory\n'; free -m 2>&1 || true; ulimit -a 2>&1 || true
        printf 'storage\n'; df -h "$PROJECT_ROOT" "${APPTAINER_TMPDIR:-$PROJECT_ROOT}" 2>&1 || true; df -i "$PROJECT_ROOT" "${APPTAINER_TMPDIR:-$PROJECT_ROOT}" 2>&1 || true
        command -v quota >/dev/null 2>&1 && timeout 10 quota -s 2>&1 || true
        command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=index,uuid,name,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>&1 || true
        command -v sacct >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]] && sacct -j "$SLURM_JOB_ID" --format=JobID,State,ExitCode,Elapsed,MaxRSS,AllocTRES -P -n 2>&1 || true
    } >> "$RUNTIME_DIAGNOSTICS_LOG"
}

runtime_diag_start() {
    RUNTIME_DIAGNOSTICS_LOG=${1:?diagnostic log required}
    local interval=${2:-60} target=${3:-}
    mkdir -p "$(dirname -- "$RUNTIME_DIAGNOSTICS_LOG")"
    : >> "$RUNTIME_DIAGNOSTICS_LOG"
    RUNTIME_DIAGNOSTICS_TARGET_PID=$target
    runtime_diag_snapshot start "$target"
    export RUNTIME_DIAGNOSTICS_LOG RUNTIME_DIAGNOSTICS_TARGET_PID
    export -f runtime_diag_snapshot runtime_diag_loop
    setsid bash -c 'runtime_diag_loop "$1" "$2"' _ "$interval" "$target" &
    RUNTIME_DIAGNOSTICS_PID=$!
}

runtime_diag_loop() {
    local interval=$1 target=${2:-} timer=
    stop() { [[ -z "$timer" ]] || kill -TERM "$timer" 2>/dev/null || true; exit 0; }
    trap stop TERM INT
    while :; do
        sleep "$interval" & timer=$!
        wait "$timer" || exit 0
        timer=
        runtime_diag_snapshot sample "$target"
    done
}

runtime_diag_note_signal() { [[ -z "${RUNTIME_DIAGNOSTICS_LOG:-}" ]] || runtime_diag_snapshot "signal-$1"; }

runtime_diag_stop() {
    local pid=${RUNTIME_DIAGNOSTICS_PID:-}
    if [[ -n "$pid" ]]; then
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
        kill -0 "$pid" 2>/dev/null && kill -KILL -- "-$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
    [[ -z "${RUNTIME_DIAGNOSTICS_LOG:-}" ]] || printf 'diagnostic utc=%s label=stop\n' "$(date -u +%FT%TZ)" >> "$RUNTIME_DIAGNOSTICS_LOG"
    RUNTIME_DIAGNOSTICS_PID=
}
