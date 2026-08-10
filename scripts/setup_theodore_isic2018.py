"""Download three pinned ISIC 2018 checkpoints from Theodore Ioannidis' Space."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

from _project_paths import REPO_ROOT
sys.path.insert(0, str(REPO_ROOT / "src"))
from thesis_fitzpatrick.datasets import download_resumable  # noqa: E402


MODEL_REVISION = "171a6dc86ee73faae0fdbda0f157d92bdb1c6596"
BASE_URL = (
    "https://huggingface.co/spaces/theodore-ioann/Skin-Lesion-Segmentation/resolve/"
    f"{MODEL_REVISION}"
)
CHECKPOINTS = {
    "unet.pt": "d58601e6433b7ebeab5ec4249ca02e413b62b28cd9e69b1719a368cb6deae5bb",
    "inception.pt": "18c17a1d87be5906b31a76558132e3c3fc16e643747b8e0859c25cb914eadce9",
    "segformer.pt": "0dcb4e5c9d19ab4caffaa324e4cf26090bcc9d37fb47840e38d81268691d8341",
}
CHECKPOINT_DIR = REPO_ROOT / "models" / "theodore-isic2018" / "checkpoints"


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
    print("U-Net, Inception y SegFormer ISIC 2018 están listos para prueba CPU.")


if __name__ == "__main__":
    main()
