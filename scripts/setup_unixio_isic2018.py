"""Download the three pinned Unixio/BertinAm ISIC 2018 checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

from _project_paths import REPO_ROOT
sys.path.insert(0, str(REPO_ROOT / "src"))
from thesis_fitzpatrick.datasets import download_resumable  # noqa: E402


MODEL_REVISION = "2a8b59e2de58d87370a92956cdea250ddf95531d"
BASE_URL = (
    "https://huggingface.co/unixio/unet-skin-lesion-segmentation/resolve/"
    f"{MODEL_REVISION}"
)
CHECKPOINTS = {
    "best_unet.pt": "99bf0a32d36ac28a42db1299af6a0ad4d562efdf685cf3fcdccd3ae8f94d1448",
    "best_unetpp.pt": "d2ba5876ffd449f426205d8498b1165618ee0159a286bfa701ff2492544ca269",
    "best_attention_unet.pt": "0d41ac498ac44d2b07ea88ef75f92370ec5312d7e6d32c39f9f0a0ce12b4812b",
}
CHECKPOINT_DIR = REPO_ROOT / "models" / "unixio-isic2018" / "checkpoints"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str, expected_sha: str) -> None:
    destination = CHECKPOINT_DIR / name
    if destination.is_file() and sha256(destination) == expected_sha:
        print(f"{name} ya está descargado y verificado.")
        return
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{name}?download=true"
    print(f"Descargando {name} desde la revisión fijada {MODEL_REVISION}...", flush=True)
    download_resumable(url, destination, expected_sha)
    print(f"Checkpoint {name} verificado: {expected_sha}")


def main() -> None:
    for name, expected_sha in CHECKPOINTS.items():
        download(name, expected_sha)
    print("U-Net, U-Net++ y Attention U-Net ISIC 2018 están listos para prueba CPU.")


if __name__ == "__main__":
    main()
