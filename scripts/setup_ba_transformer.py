"""Install pinned BA-Transformer source and verified ISIC 2016 checkpoint."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from _project_paths import REPO_ROOT


SOURCE_REPOSITORY = "jcwang123/BA-Transformer"
SOURCE_COMMIT = "d1ccdff68beac82d18c2cd64bb800e51d7afc5e1"
WEIGHTS_URL = "https://drive.google.com/uc?id=1-eMHYX1fr-QvI3n50S0xqWcxc3FGsMgE"
ARCHIVE_SHA256 = "0d837469bfbc7306d1fc61d1f4292243d4447a0f45d2ddc850f7947932aec88e"
CHECKPOINT_SHA256 = "62b4148b26b01b0b17b4125d74115ed49c507eab4d39ec8ff8e063a2f7980233"
CHECKPOINT_MEMBER = "isic2016/bat_1_1_0_e6_loss_0_aug_1/fold_0/model/best.pkl"
MODEL_ROOT = REPO_ROOT / "models" / "ba-transformer"
SOURCE_DIR = MODEL_ROOT / "source"
CHECKPOINT_DIR = MODEL_ROOT / "checkpoints"
ARCHIVE_PATH = CHECKPOINT_DIR / "published_weights.zip"
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_isic2016.pkl"


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
            raise SystemExit("No se encontró GitHub CLI (gh).")
        SOURCE_DIR.parent.mkdir(parents=True, exist_ok=True)
        run([gh, "repo", "clone", SOURCE_REPOSITORY, str(SOURCE_DIR), "--", "--depth", "1"])
    current = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=SOURCE_DIR, check=True, text=True, capture_output=True
    ).stdout.strip()
    if current != SOURCE_COMMIT:
        run(["git", "fetch", "--depth", "1", "origin", SOURCE_COMMIT], cwd=SOURCE_DIR)
        run(["git", "checkout", "--detach", SOURCE_COMMIT], cwd=SOURCE_DIR)
    print(f"Código BA-Transformer fijado en {SOURCE_COMMIT}.")


def install_checkpoint() -> None:
    if CHECKPOINT_PATH.is_file() and sha256(CHECKPOINT_PATH) == CHECKPOINT_SHA256:
        ARCHIVE_PATH.unlink(missing_ok=True)
        print("Checkpoint BA-Transformer ya descargado y verificado.")
        return
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "gdown", WEIGHTS_URL, "-O", str(ARCHIVE_PATH)])
    archive_hash = sha256(ARCHIVE_PATH)
    if archive_hash != ARCHIVE_SHA256:
        raise SystemExit(
            f"Archivo ZIP inválido: esperado {ARCHIVE_SHA256}, obtenido {archive_hash}."
        )
    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        member = archive.getinfo(CHECKPOINT_MEMBER)
        with archive.open(member) as source, CHECKPOINT_PATH.open("wb") as destination:
            shutil.copyfileobj(source, destination)
    checkpoint_hash = sha256(CHECKPOINT_PATH)
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise SystemExit(
            f"Checkpoint inválido: esperado {CHECKPOINT_SHA256}, obtenido {checkpoint_hash}."
        )
    ARCHIVE_PATH.unlink()
    print(f"Checkpoint ISIC 2016 verificado: {CHECKPOINT_SHA256}")


def main() -> None:
    install_source()
    install_checkpoint()
    print("BA-Transformer está listo para la prueba CPU.")


if __name__ == "__main__":
    main()
