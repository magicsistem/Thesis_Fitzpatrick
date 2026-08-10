#!/usr/bin/env bash
# Read-only environment report for a CEDIA Open OnDemand terminal.
set -u

section() { printf '\n==== %s ====\n' "$1"; }
optional() {
    printf '$'
    printf ' %q' "$@"
    printf '\n'
    "$@" 2>&1 || printf '[not available or restricted; exit=%s]\n' "$?"
}

section "Identity"
optional hostname
optional date --iso-8601=seconds
optional id
printf 'USER=%s\nHOME=%s\nPWD=%s\n' "${USER:-unknown}" "${HOME:-unknown}" "$PWD"

section "HOME storage and quota"
optional df -h "${HOME:-.}"
optional df -i "${HOME:-.}"
if command -v quota >/dev/null 2>&1; then optional quota -s; else echo "quota: command unavailable"; fi
if command -v lfs >/dev/null 2>&1; then optional lfs quota -u "${USER:-}" "${HOME:-.}"; else echo "lfs quota: command unavailable"; fi

section "Visible SLURM configuration"
if command -v sinfo >/dev/null 2>&1; then optional sinfo -o '%P %a %l %D %G %c %m'; else echo "sinfo: command unavailable"; fi
if command -v scontrol >/dev/null 2>&1; then
    optional scontrol show partition gpu
    optional scontrol show config
else
    echo "scontrol: command unavailable"
fi
if command -v squeue >/dev/null 2>&1; then optional squeue -u "${USER:-}"; else echo "squeue: command unavailable"; fi
optional bash -c 'ulimit -a'

section "Host tools"
for tool in gcc g++ make nvcc; do
    if command -v "$tool" >/dev/null 2>&1; then
        printf '%s: %s\n' "$tool" "$(command -v "$tool")"
        "$tool" --version 2>&1 | head -n 3 || true
    else
        printf '%s: unavailable\n' "$tool"
    fi
done

runtime=""
for candidate in apptainer singularity; do
    if command -v "$candidate" >/dev/null 2>&1; then runtime="$candidate"; break; fi
done
section "Container runtime"
if [[ -n "$runtime" ]]; then optional "$runtime" version; else echo "Neither apptainer nor singularity is available in PATH."; fi

section "Limited SIF search"
declare -a candidates=()
if [[ -n "${SIF_PATH:-}" ]]; then
    candidates+=("$SIF_PATH")
else
    search_roots=("${HOME:-.}")
    if [[ -n "${SIF_SEARCH_ROOTS:-}" ]]; then
        IFS=: read -r -a extra_roots <<< "$SIF_SEARCH_ROOTS"
        search_roots+=("${extra_roots[@]}")
    fi
    for root in "${search_roots[@]}"; do
        if [[ -d "$root" && -r "$root" ]]; then
            printf 'Searching %s (max depth 6)\n' "$root"
            while IFS= read -r -d '' found; do candidates+=("$found"); done \
                < <(find "$root" -maxdepth 6 -type f -name 'pytorch_24.01-py3.sif' -print0 2>/dev/null)
        else
            printf 'Skipped inaccessible search root: %s\n' "$root"
        fi
    done
fi

if ((${#candidates[@]} == 0)); then
    cat <<'EOF'
SIF not found. Re-run with the path observed in CEDIA:
SIF_PATH="/ruta/observada/en/CEDIA/pytorch_24.01-py3.sif" \
bash scripts/hpc/inspect_cedia_environment.sh
EOF
    exit 0
fi

printf 'Candidates found:\n'
printf '  %s\n' "${candidates[@]}"
if ((${#candidates[@]} != 1)); then
    echo "More than one candidate found; set SIF_PATH explicitly before inspecting an image."
    exit 0
fi
sif=${candidates[0]}
if [[ ! -f "$sif" || ! -r "$sif" ]]; then
    echo "SIF_PATH is not a readable regular file: $sif"
    exit 0
fi

section "SIF identity"
optional stat -c 'path=%n permissions=%A owner=%U:%G bytes=%s modified=%y' "$sif"
if command -v sha256sum >/dev/null 2>&1; then optional sha256sum "$sif"; else echo "sha256sum: command unavailable"; fi
if [[ -z "$runtime" ]]; then
    echo "A runtime is required to inspect the image; file metadata above is still valid."
    exit 0
fi
optional "$runtime" inspect "$sif"

nv_args=()
gpu_session=false
if [[ -n "${SLURM_JOB_ID:-}" || -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        gpu_session=true
        nv_args=(--nv)
    fi
fi

section "GPU visibility"
printf 'SLURM_JOB_ID=%s\nCUDA_VISIBLE_DEVICES=%s\n' "${SLURM_JOB_ID:-not-set}" "${CUDA_VISIBLE_DEVICES:-not-set}"
if $gpu_session; then optional nvidia-smi; else echo "No allocated GPU is visible; GPU compatibility is not evaluated in this terminal."; fi

container_python=$(
    "$runtime" exec "${nv_args[@]}" "$sif" sh -c 'command -v python3 || command -v python' 2>/dev/null | head -n 1
)
section "Container Python and packages"
if [[ -z "$container_python" ]]; then
    echo "No python/python3 executable found inside the image."
else
    optional "$runtime" exec "${nv_args[@]}" "$sif" "$container_python" --version
    optional "$runtime" exec "${nv_args[@]}" "$sif" "$container_python" -m pip --version
    "$runtime" exec "${nv_args[@]}" "$sif" "$container_python" - <<'PY' 2>&1 || true
import importlib
import platform

print("platform:", platform.platform())
packages = (
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("scikit-image", "skimage"),
    ("OpenCV", "cv2"),
    ("timm", "timm"),
    ("einops", "einops"),
    ("Pillow", "PIL"),
    ("PyYAML", "yaml"),
    ("transformers", "transformers"),
    ("segmentation-models-pytorch", "segmentation_models_pytorch"),
)
for label, module_name in packages:
    try:
        module = importlib.import_module(module_name)
        print(f"{label}: {getattr(module, '__version__', 'installed/version unavailable')}")
    except Exception as exc:
        print(f"{label}: unavailable ({type(exc).__name__}: {exc})")
try:
    import torch
    print("PyTorch:", torch.__version__)
    try:
        import torchvision
        print("torchvision:", torchvision.__version__)
    except Exception as exc:
        print(f"torchvision: unavailable ({type(exc).__name__}: {exc})")
    print("torch CUDA build:", torch.version.cuda)
    print("cuDNN:", torch.backends.cudnn.version())
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            print(f"CUDA device {index}:", torch.cuda.get_device_name(index))
except Exception as exc:
    print(f"PyTorch: unavailable ({type(exc).__name__}: {exc})")
PY
fi

section "Container compilers"
for tool in gcc g++ make nvcc; do
    printf '%s:\n' "$tool"
    "$runtime" exec "${nv_args[@]}" "$sif" sh -c \
        "if command -v '$tool' >/dev/null 2>&1; then command -v '$tool'; '$tool' --version 2>&1 | head -n 3; else echo unavailable; fi" 2>&1 || true
done

section "Conclusion"
echo "This report inventories the current portal environment only. It does not certify benchmark compatibility."
