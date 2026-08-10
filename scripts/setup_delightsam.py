"""Install pinned De-LightSAM source and its official dermoscopy checkpoint."""

from __future__ import annotations

import shutil
import hashlib
from pathlib import Path
import subprocess
import sys
import zipfile

from _project_paths import REPO_ROOT


SOURCE_REPOSITORY = "xq141839/De-LightSAM"
SOURCE_COMMIT = "d2260d75e4ad1f5da8e053711cd9c36128088a7b"
WEIGHTS_URL = "https://drive.google.com/uc?id=1kikT1Sjp6TBJQqBJgM80dP2nSTf2PpJM"
ARCHIVE_SHA256 = "837e3f4654abe849aa634be2da3864b09d3b2a22f16b8178815afede187b0c9f"
CHECKPOINT_SHA256 = "79555730abffb5b39bef5d59afb7b89b13468348b77efa144649b90fddc01778"
CHECKPOINT_MEMBER = "ESP-MedSAM/ESP_isic_best.pth"
MODEL_ROOT = REPO_ROOT / "models" / "delightsam"
SOURCE_DIR = MODEL_ROOT / "source"
CHECKPOINT_DIR = MODEL_ROOT / "checkpoints"
ARCHIVE_PATH = CHECKPOINT_DIR / "published_model_zoo.zip"
CHECKPOINT_PATH = CHECKPOINT_DIR / "ESP_isic_best.pth"


def run(command: list[str], *, cwd: Path = REPO_ROOT) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_source() -> None:
    if SOURCE_DIR.exists():
        if not (SOURCE_DIR / ".git").is_dir():
            raise SystemExit(f"{SOURCE_DIR} existe, pero no es un clon Git.")
    else:
        SOURCE_DIR.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--no-checkout", f"https://github.com/{SOURCE_REPOSITORY}.git", str(SOURCE_DIR)])
    current = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=SOURCE_DIR, check=True, text=True, capture_output=True
    ).stdout.strip()
    if current != SOURCE_COMMIT:
        run(["git", "fetch", "--depth", "1", "origin", SOURCE_COMMIT], cwd=SOURCE_DIR)
        run(["git", "checkout", "--detach", SOURCE_COMMIT], cwd=SOURCE_DIR)
    print(f"Código De-LightSAM fijado en {SOURCE_COMMIT}.")


def install_checkpoint() -> None:
    if CHECKPOINT_PATH.is_file() and sha256(CHECKPOINT_PATH) == CHECKPOINT_SHA256:
        ARCHIVE_PATH.unlink(missing_ok=True)
        print("Checkpoint De-LightSAM Dermoscopy ya descargado y verificado.")
        return
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "gdown", "--continue", WEIGHTS_URL, "-O", str(ARCHIVE_PATH)])
    archive_hash = sha256(ARCHIVE_PATH)
    if archive_hash != ARCHIVE_SHA256:
        raise SystemExit(
            f"ZIP De-LightSAM inválido: esperado {ARCHIVE_SHA256}, obtenido {archive_hash}."
        )
    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        with archive.open(CHECKPOINT_MEMBER) as source, CHECKPOINT_PATH.open("wb") as output:
            shutil.copyfileobj(source, output)
    checkpoint_hash = sha256(CHECKPOINT_PATH)
    if checkpoint_hash != CHECKPOINT_SHA256:
        CHECKPOINT_PATH.unlink(missing_ok=True)
        raise SystemExit(
            f"Checkpoint inválido: esperado {CHECKPOINT_SHA256}, obtenido {checkpoint_hash}."
        )
    ARCHIVE_PATH.unlink()
    print(f"Checkpoint Dermoscopy verificado: {CHECKPOINT_SHA256}")


def main() -> None:
    install_source()
    install_checkpoint()
    print("De-LightSAM Dermoscopy está listo; CPU será lento por su entrada 1024×1024.")


if __name__ == "__main__":
    main()
