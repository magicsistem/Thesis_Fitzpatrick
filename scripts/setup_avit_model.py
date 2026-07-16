"""Install the pinned AViT source and verified ISIC 2018 checkpoint."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

from _project_paths import REPO_ROOT


SOURCE_REPOSITORY = "siyi-wind/AViT"
SOURCE_COMMIT = "b77b01af263727b2b2cf2555a5ed9f1f48a2b4a2"
WEIGHTS_FOLDER = "https://drive.google.com/drive/folders/1ct8_GztVLC5BCWpzGxp9MZKP5ZQk9Cve?usp=sharing"
CHECKPOINT_NAME = "AViT_ViT_B_ISIC_best.pth"
CHECKPOINT_SHA256 = "9b4ad401483d96535769433f4781da42179bec6a7ef932a1d02a7f786e9f24db"
MODEL_ROOT = REPO_ROOT / "models" / "avit"
SOURCE_DIR = MODEL_ROOT / "source"
CHECKPOINT_DIR = MODEL_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / CHECKPOINT_NAME


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
        gh = shutil.which("gh")
        if not gh:
            raise SystemExit("No se encontró gh. Instala y autentica GitHub CLI primero.")
        SOURCE_DIR.parent.mkdir(parents=True, exist_ok=True)
        run([gh, "repo", "clone", SOURCE_REPOSITORY, str(SOURCE_DIR), "--", "--depth", "1"])

    current = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=SOURCE_DIR, check=True, text=True, capture_output=True
    ).stdout.strip()
    if current != SOURCE_COMMIT:
        run(["git", "fetch", "--depth", "1", "origin", SOURCE_COMMIT], cwd=SOURCE_DIR)
        run(["git", "checkout", "--detach", SOURCE_COMMIT], cwd=SOURCE_DIR)
    print(f"Código AViT fijado en {SOURCE_COMMIT}.")


def install_checkpoint() -> None:
    if CHECKPOINT_PATH.is_file() and sha256(CHECKPOINT_PATH) == CHECKPOINT_SHA256:
        print("Checkpoint AViT ya descargado y verificado.")
        return
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable,
        "-m",
        "gdown",
        "--folder",
        WEIGHTS_FOLDER,
        "-O",
        str(CHECKPOINT_DIR),
    ])
    if not CHECKPOINT_PATH.is_file():
        raise SystemExit(f"La descarga no produjo {CHECKPOINT_PATH}.")
    actual = sha256(CHECKPOINT_PATH)
    if actual != CHECKPOINT_SHA256:
        raise SystemExit(
            "Checkpoint AViT inválido: "
            f"SHA-256 esperado {CHECKPOINT_SHA256}, obtenido {actual}."
        )
    print(f"Checkpoint verificado: {CHECKPOINT_SHA256}")


def main() -> None:
    install_source()
    install_checkpoint()
    print("AViT está listo para la prueba de inferencia.")


if __name__ == "__main__":
    main()

