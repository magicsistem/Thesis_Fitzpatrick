"""Download the pinned and verified U-Net/ResNet34 ISIC 2018 checkpoint."""

from __future__ import annotations

import hashlib
from pathlib import Path
import urllib.request

from _project_paths import REPO_ROOT


MODEL_REVISION = "03d64aeb97ef50ae1b3a68224645c7eeb081a61a"
CHECKPOINT_URL = (
    "https://huggingface.co/Jesusrodriguezf90/"
    "unet-resnet34-isic2018-segmentation/resolve/"
    f"{MODEL_REVISION}/best_unet_resnet34.pth?download=true"
)
CHECKPOINT_SHA256 = "1ea87e341552768234b367c3b68704030bc1bc08991c323508ea5c5086d9d334"
CHECKPOINT_PATH = REPO_ROOT / "models" / "unet-resnet34" / "checkpoints" / "best_unet_resnet34.pth"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_checkpoint() -> None:
    if CHECKPOINT_PATH.is_file() and sha256(CHECKPOINT_PATH) == CHECKPOINT_SHA256:
        print("Checkpoint U-Net/ResNet34 ya descargado y verificado.")
        return
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = CHECKPOINT_PATH.with_suffix(".download")
    temporary.unlink(missing_ok=True)
    print(f"Descargando checkpoint fijado en {MODEL_REVISION}...", flush=True)
    try:
        with urllib.request.urlopen(CHECKPOINT_URL) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    actual_sha = sha256(temporary)
    if actual_sha != CHECKPOINT_SHA256:
        temporary.unlink(missing_ok=True)
        raise SystemExit(
            f"Checkpoint inválido: esperado {CHECKPOINT_SHA256}, obtenido {actual_sha}."
        )
    temporary.replace(CHECKPOINT_PATH)
    print(f"Checkpoint ISIC 2018 verificado: {CHECKPOINT_SHA256}")


def main() -> None:
    download_checkpoint()
    print("U-Net/ResNet34 está listo para la prueba CPU.")


if __name__ == "__main__":
    main()
