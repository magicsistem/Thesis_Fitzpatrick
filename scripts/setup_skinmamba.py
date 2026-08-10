"""Install pinned SkinMamba source and both published checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from _project_paths import REPO_ROOT


SOURCE_REPOSITORY = "zs1314/SkinMamba"
SOURCE_COMMIT = "131a85da14da4d3bf135b52912d274191c05f3b8"
WEIGHTS_URL = "https://drive.google.com/uc?id=1ialdv8WJoKEZkkwiqGuleEWnXOyyUEoW"
ARCHIVE_SHA256 = "95cda03c7952a3c7a984dbd4403ab7f821ff10277939554f52d49a7fe28fb4b0"
CHECKPOINTS = {
    "isic17.pth": "9b941f1459dd6e7660bd5ec32b2b58ba861c596ac8b14708a43a9c940f0645a7",
    "isic18.pth": "36a2c352cd39011db3f416373be1bdd98b079d031fc778dbc640780823388543",
}
MODEL_ROOT = REPO_ROOT / "models" / "skinmamba"
SOURCE_DIR = MODEL_ROOT / "source"
CHECKPOINT_DIR = MODEL_ROOT / "checkpoints"
ARCHIVE_PATH = CHECKPOINT_DIR / "published_weights.zip"


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
    print(f"Código SkinMamba fijado en {SOURCE_COMMIT}.")


def install_checkpoints() -> None:
    if all(
        (CHECKPOINT_DIR / name).is_file() and sha256(CHECKPOINT_DIR / name) == expected
        for name, expected in CHECKPOINTS.items()
    ):
        ARCHIVE_PATH.unlink(missing_ok=True)
        print("Checkpoints SkinMamba ya descargados y verificados.")
        return
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "gdown", WEIGHTS_URL, "-O", str(ARCHIVE_PATH)])
    actual_archive = sha256(ARCHIVE_PATH)
    if actual_archive != ARCHIVE_SHA256:
        raise SystemExit(
            f"ZIP SkinMamba inválido: esperado {ARCHIVE_SHA256}, obtenido {actual_archive}."
        )
    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        for name, expected in CHECKPOINTS.items():
            destination = CHECKPOINT_DIR / name
            with archive.open(f"weight/{name}") as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)
            actual = sha256(destination)
            if actual != expected:
                destination.unlink(missing_ok=True)
                raise SystemExit(
                    f"Checkpoint {name} inválido: esperado {expected}, obtenido {actual}."
                )
            print(f"Checkpoint {name} verificado: {expected}")
    ARCHIVE_PATH.unlink()


def main() -> None:
    install_source()
    install_checkpoints()
    print("SkinMamba ISIC 2017 e ISIC 2018 están listos para prueba CPU.")


if __name__ == "__main__":
    main()
