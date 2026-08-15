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
        runtime_diag_darknet_state
        printf 'cgroup\n'; cat /proc/self/cgroup 2>&1 || true
        while IFS= read -r item; do
            [[ -r "$item" ]] && { printf '%s=' "$item"; cat "$item"; }
        done < <(find /sys/fs/cgroup -maxdepth 3 -type f \( -name memory.current -o -name memory.events -o -name memory.max -o -name pids.current -o -name pids.events \) 2>/dev/null | head -40)
        printf 'memory\n'; free -m 2>&1 || true; ulimit -a 2>&1 || true
        printf 'storage\n'; df -h "$PROJECT_ROOT" "${APPTAINER_TMPDIR:-$PROJECT_ROOT}" 2>&1 || true; df -i "$PROJECT_ROOT" "${APPTAINER_TMPDIR:-$PROJECT_ROOT}" 2>&1 || true
        # Optional NFS quota queries can block; diagnostics must never retain
        # the job after a failed payload or a signal.
        command -v quota >/dev/null 2>&1 && timeout 1 quota -s 2>&1 || true
        runtime_diag_gpu
        command -v sacct >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]] && sacct -j "$SLURM_JOB_ID" --format=JobID,State,ExitCode,Elapsed,MaxRSS,AllocTRES -P -n 2>&1 || true
    } >> "$RUNTIME_DIAGNOSTICS_LOG"
}

runtime_diag_json_number() {
    local key=$1 state=${RUNTIME_DIAGNOSTICS_STATE_PATH:-}
    [[ -r "$state" ]] || return 0
    sed -n "s/.*\"$key\"[[:space:]]*:[[:space:]]*\([-0-9][0-9]*\).*/\1/p" "$state" | head -1
}

runtime_diag_json_string() {
    local key=$1 state=${RUNTIME_DIAGNOSTICS_STATE_PATH:-}
    [[ -r "$state" ]] || return 0
    sed -n "s/.*\"$key\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/p" "$state" | head -1
}

runtime_diag_darknet_state() {
    local state=${RUNTIME_DIAGNOSTICS_STATE_PATH:-} pid pgid attempt observed started stopped rc log live_iteration
    printf 'darknet_state path=%s\n' "$state"
    [[ -r "$state" ]] || return 0
    pid=$(runtime_diag_json_number darknet_pid); pgid=$(runtime_diag_json_number darknet_pgid)
    attempt=$(runtime_diag_json_number attempt); observed=$(runtime_diag_json_number observed_iteration)
    started=$(runtime_diag_json_string darknet_started_utc); stopped=$(runtime_diag_json_string darknet_stopped_utc); rc=$(runtime_diag_json_number returncode)
    log=$(runtime_diag_json_string log_path)
    if [[ -r "$log" ]]; then
        live_iteration=$(awk -F: '/^[0-9]+:/ {if ($1 > max) max=$1} END {print max+0}' "$log" 2>/dev/null || true)
    else
        live_iteration=0
    fi
    printf 'darknet pid=%s pgid=%s attempt=%s observed_iteration=%s live_iteration=%s started_utc=%s stopped_utc=%s returncode=%s\n' "$pid" "$pgid" "$attempt" "$observed" "$live_iteration" "$started" "$stopped" "$rc"
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] && ps -o pid=,ppid=,pgid=,sid=,stat=,etime=,pcpu=,rss=,args= -p "$pid" 2>&1 || true
}

runtime_diag_gpu() {
    local utc
    utc=$(date -u +%FT%TZ)
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,uuid,name,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>&1 || true
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | \
            awk -F, -v utc="$utc" '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); gsub(/^[[:space:]]+|[[:space:]]+$/, "", $3); gsub(/^[[:space:]]+|[[:space:]]+$/, "", $4); gsub(/^[[:space:]]+|[[:space:]]+$/, "", $5); printf "diagnostic_gpu utc=%s index=%s util=%s memory_mb=%s power_w=%s temp_c=%s\\n", utc,$1,$2,$3,$4,$5}' || true
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1 || true
    else
        printf 'diagnostic_gpu unavailable\n'
    fi
}

runtime_diag_gpu_summary() {
    local log=${RUNTIME_DIAGNOSTICS_LOG:-}
    [[ -r "$log" ]] || return 0
    awk '
      /^diagnostic_gpu / { for (i=1;i<=NF;i++) if ($i ~ /^util=/) { split($i,a,"="); if (a[2] ~ /^[0-9.]+$/) v[++n]=a[2]+0 } }
      END { if (!n) exit; for(i=1;i<=n;i++){sum+=v[i]; if(v[i]==0) zero++; if(v[i]>max)max=v[i]}; for(i=1;i<=n;i++)for(j=i+1;j<=n;j++)if(v[j]<v[i]){t=v[i];v[i]=v[j];v[j]=t}; mid=v[int((n+1)/2)]; p95=v[int((95*n+99)/100)]; printf "diagnostic_gpu_summary samples=%d mean=%.3f median=%.3f p95=%.3f max=%.3f zero_util_fraction=%.3f\\n",n,sum/n,mid,p95,max,zero/n }' "$log" >> "$log"
}

runtime_diag_start() {
    RUNTIME_DIAGNOSTICS_LOG=${1:?diagnostic log required}
    local interval=${2:-60} target=${3:-}
    mkdir -p "$(dirname -- "$RUNTIME_DIAGNOSTICS_LOG")"
    : >> "$RUNTIME_DIAGNOSTICS_LOG"
    RUNTIME_DIAGNOSTICS_TARGET_PID=$target
    runtime_diag_snapshot start "$target"
    export RUNTIME_DIAGNOSTICS_LOG RUNTIME_DIAGNOSTICS_TARGET_PID
    export -f runtime_diag_snapshot runtime_diag_loop runtime_diag_json_number runtime_diag_json_string runtime_diag_darknet_state runtime_diag_gpu
    RUNTIME_DIAGNOSTICS_MONITOR_STATE=$(mktemp "$(dirname -- "$RUNTIME_DIAGNOSTICS_LOG")/.runtime-monitor.XXXXXX")
    export RUNTIME_DIAGNOSTICS_MONITOR_STATE
    setsid bash -c 'runtime_diag_loop "$1" "$2" "$3"' _ "$interval" "$target" "$RUNTIME_DIAGNOSTICS_MONITOR_STATE" &
    RUNTIME_DIAGNOSTICS_PID=$!
    RUNTIME_DIAGNOSTICS_PGID=$(ps -o pgid= -p "$RUNTIME_DIAGNOSTICS_PID" 2>/dev/null | tr -d ' ' || true)
    RUNTIME_DIAGNOSTICS_SID=$(ps -o sid= -p "$RUNTIME_DIAGNOSTICS_PID" 2>/dev/null | tr -d ' ' || true)
    if ! runtime_diag_monitor_isolated; then
        printf 'diagnostic utc=%s label=monitor-rejected pid=%s pgid=%s sid=%s\n' "$(date -u +%FT%TZ)" "$RUNTIME_DIAGNOSTICS_PID" "$RUNTIME_DIAGNOSTICS_PGID" "$RUNTIME_DIAGNOSTICS_SID" >> "$RUNTIME_DIAGNOSTICS_LOG"
        kill -TERM "$RUNTIME_DIAGNOSTICS_PID" 2>/dev/null || true
        wait "$RUNTIME_DIAGNOSTICS_PID" 2>/dev/null || true
        rm -f -- "$RUNTIME_DIAGNOSTICS_MONITOR_STATE"
        RUNTIME_DIAGNOSTICS_PID= RUNTIME_DIAGNOSTICS_PGID= RUNTIME_DIAGNOSTICS_SID= RUNTIME_DIAGNOSTICS_MONITOR_STATE=
    fi
}

runtime_diag_loop() {
    local interval=$1 target=${2:-} state_file=${3:?state required} timer=
    stop() { [[ -z "$timer" ]] || kill -TERM "$timer" 2>/dev/null || true; [[ -z "$timer" ]] || wait "$timer" 2>/dev/null || true; rm -f -- "$state_file"; exit 0; }
    trap stop TERM INT
    while :; do
        sleep "$interval" & timer=$!; printf '%s\n' "$timer" > "$state_file"
        wait "$timer" || exit 0
        timer=; : > "$state_file"
        runtime_diag_snapshot sample "$target"
    done
}

runtime_diag_note_signal() { [[ -z "${RUNTIME_DIAGNOSTICS_LOG:-}" ]] || runtime_diag_snapshot "signal-$1"; }

runtime_diag_monitor_isolated() {
    local pid=${RUNTIME_DIAGNOSTICS_PID:-} pgid=${RUNTIME_DIAGNOSTICS_PGID:-} sid=${RUNTIME_DIAGNOSTICS_SID:-} target=${RUNTIME_DIAGNOSTICS_TARGET_PID:-}
    [[ "$pid" =~ ^[2-9][0-9]*$|^[1-9][0-9]{2,}$ ]] || return 1
    [[ "$pgid" == "$pid" && "$sid" == "$pid" ]] || return 1
    [[ "$pid" != "$$" && "$pid" != "$PPID" && "$pid" != "$target" ]] || return 1
    [[ -z "$target" || "$pgid" != "$(ps -o pgid= -p "$target" 2>/dev/null | tr -d ' ' || true)" ]] || return 1
    kill -0 "$pid" 2>/dev/null
}

runtime_diag_timer_is_child() {
    local timer=${1:-} pid=${RUNTIME_DIAGNOSTICS_PID:-}
    [[ "$timer" =~ ^[2-9][0-9]*$|^[1-9][0-9]{2,}$ && "$pid" =~ ^[2-9][0-9]*$|^[1-9][0-9]{2,}$ ]] || return 1
    [[ "$(ps -o ppid= -p "$timer" 2>/dev/null | tr -d ' ')" == "$pid" ]]
}

runtime_diag_stop() {
    local pid=${RUNTIME_DIAGNOSTICS_PID:-} timer= state=${RUNTIME_DIAGNOSTICS_MONITOR_STATE:-}
    if runtime_diag_monitor_isolated; then
        [[ -r "$state" ]] && timer=$(head -n 1 "$state" || true)
        runtime_diag_timer_is_child "$timer" && kill -TERM "$timer" 2>/dev/null || true
        kill -TERM "$pid" 2>/dev/null || true
        for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
        if kill -0 "$pid" 2>/dev/null; then
            runtime_diag_timer_is_child "$timer" && kill -KILL "$timer" 2>/dev/null || true
            kill -KILL "$pid" 2>/dev/null || true
        fi
        wait "$pid" 2>/dev/null || true
    elif [[ -n "$pid" && -n "${RUNTIME_DIAGNOSTICS_LOG:-}" ]]; then
        printf 'diagnostic utc=%s label=monitor-stop-refused pid=%s pgid=%s sid=%s\n' "$(date -u +%FT%TZ)" "$pid" "${RUNTIME_DIAGNOSTICS_PGID:-}" "${RUNTIME_DIAGNOSTICS_SID:-}" >> "$RUNTIME_DIAGNOSTICS_LOG"
    fi
    [[ -z "$state" ]] || rm -f -- "$state"
    [[ -z "${RUNTIME_DIAGNOSTICS_LOG:-}" ]] || { runtime_diag_snapshot stop "${RUNTIME_DIAGNOSTICS_TARGET_PID:-}"; runtime_diag_gpu_summary; printf 'diagnostic utc=%s label=stop\n' "$(date -u +%FT%TZ)" >> "$RUNTIME_DIAGNOSTICS_LOG"; }
    RUNTIME_DIAGNOSTICS_PID= RUNTIME_DIAGNOSTICS_PGID= RUNTIME_DIAGNOSTICS_SID= RUNTIME_DIAGNOSTICS_MONITOR_STATE=
}
