#!/usr/bin/env bash
# Internal stage: called once through run_in_container.sh to avoid repeated SIF conversion.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd -P)
VENV_PATH=${VENV_PATH:-$PROJECT_ROOT/.cedia/venv}
MODE=${1:-models}
[[ "$MODE" == dependencies || "$MODE" == models ]] || { echo "Mode must be dependencies or models" >&2; exit 2; }

if [[ ! -x "$VENV_PATH/bin/python" ]]; then
    python3 -m venv --system-site-packages "$VENV_PATH"
fi
source "$VENV_PATH/bin/activate"
python -m pip install --disable-pip-version-check \
    --constraint "$PROJECT_ROOT/configs/hpc/core-constraints.txt" \
    --requirement "$PROJECT_ROOT/configs/hpc/requirements-cuda.txt"

python - <<'PY'
import numpy, torch, torchvision
assert numpy.__version__ == "1.24.4", numpy.__version__
assert torch.__version__ == "2.2.0a0+81ea7a4", torch.__version__
assert torchvision.__version__ == "0.17.0a0", torchvision.__version__
print("Core SIF packages were preserved.")
PY

if [[ "$MODE" == models ]]; then
    python scripts/hpc/bootstrap_resources.py --require-sources
    setup_scripts=(
        setup_avit_model.py setup_ultralight_vm_unet.py setup_ba_transformer.py
        setup_unet_resnet34.py setup_skinmamba.py setup_unixio_isic2018.py
        setup_theodore_isic2018.py setup_vmunet.py setup_delightsam.py
    )
    for setup_script in "${setup_scripts[@]}"; do
        python "scripts/$setup_script"
    done
    python scripts/setup_yolov3_darknet.py --download-bootstrap --confirm-download \
        --expected-weights-sha256 2495c2690283e4e0bc2050cbd4660b77a8074e14e9c11150c6412fd63db496a7 \
        --build --gpu
    python scripts/hpc/bootstrap_resources.py --require-sources --require-checkpoints
fi

python -m pip check
echo "bootstrap_inside_container completed mode=$MODE"
