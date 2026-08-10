"""Install pinned UltraLight VM-UNet source and its published checkpoint."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys
import zipfile

from _project_paths import REPO_ROOT


SOURCE_REPOSITORY = "wurenkai/UltraLight-VM-UNet"
SOURCE_COMMIT = "27e44181b3cd5b0b2ab3ad1e3ddb8cad67368fbd"
WEIGHTS_URL = "https://drive.google.com/uc?id=1lbyWRlDGCl5YL65V2-OwwZui2dHH8pH7"
ARCHIVE_SHA256 = "ba51198d93f70c0281ca81d1f466c937f449b81b5fe30775a4a2bd8b0e379c99"
CHECKPOINT_SHA256 = "43b11155c19c2296707ec4dee3c417529ea0b54eb111adee864b2274ec8df52a"
MODEL_ROOT = REPO_ROOT / "models" / "ultralight-vm-unet"
SOURCE_DIR = MODEL_ROOT / "source"
CHECKPOINT_DIR = MODEL_ROOT / "checkpoints"
ARCHIVE_PATH = CHECKPOINT_DIR / "published_skin_lesion_weights.zip"
CHECKPOINT_PATH = CHECKPOINT_DIR / "UltraLight_VM_UNet.pth"


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
    print(f"Código UltraLight VM-UNet fijado en {SOURCE_COMMIT}.")


def install_checkpoint() -> None:
    if CHECKPOINT_PATH.is_file() and sha256(CHECKPOINT_PATH) == CHECKPOINT_SHA256:
        ARCHIVE_PATH.unlink(missing_ok=True)
        print("Checkpoint UltraLight VM-UNet ya descargado y verificado.")
        return
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "gdown", "--continue", WEIGHTS_URL, "-O", str(ARCHIVE_PATH)])
    archive_hash = sha256(ARCHIVE_PATH)
    if archive_hash != ARCHIVE_SHA256:
        raise SystemExit(
            f"Archivo ZIP inválido: esperado {ARCHIVE_SHA256}, obtenido {archive_hash}."
        )
    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        member = archive.getinfo("UltraLight_VM_UNet.pth")
        with archive.open(member) as source, CHECKPOINT_PATH.open("wb") as destination:
            shutil.copyfileobj(source, destination)
    checkpoint_hash = sha256(CHECKPOINT_PATH)
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise SystemExit(
            f"Checkpoint inválido: esperado {CHECKPOINT_SHA256}, obtenido {checkpoint_hash}."
        )
    ARCHIVE_PATH.unlink()
    print(f"Checkpoint verificado: {CHECKPOINT_SHA256}")


def main() -> None:
    install_source()
    install_checkpoint()
    print("UltraLight VM-UNet está listo para la prueba CPU.")


if __name__ == "__main__":
    main()
