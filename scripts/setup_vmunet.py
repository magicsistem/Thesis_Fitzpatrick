"""Install pinned VM-UNet source and both official ISIC checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

from _project_paths import REPO_ROOT


SOURCE_REPOSITORY = "JCruan519/VM-UNet"
SOURCE_COMMIT = "b87827eb5a5faff00bd4b7b6505e56d3ca813ca2"
CHECKPOINTS = {
    "best-vmunet-isic17.pth": (
        "https://drive.google.com/uc?id=1QpU9MCsACtGFKQZYHpaATt5J71b_5kxv",
        "61af065274210838ed65d165bd908a0b831ec6be8f2739c60e990fcfdd1071ce",
    ),
    "best-vmunet-isic18.pth": (
        "https://drive.google.com/uc?id=1T_g3RPO2tef6HZ2kx_ItGFnUn-tncnli",
        "a5b2c175ccb2e2fa428004a1c90023ffd80ef3a9d9c485f7f77ccbe4427abd38",
    ),
}
MODEL_ROOT = REPO_ROOT / "models" / "vmunet"
SOURCE_DIR = MODEL_ROOT / "source"
CHECKPOINT_DIR = MODEL_ROOT / "checkpoints"


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
    print(f"Código VM-UNet fijado en {SOURCE_COMMIT}.")


def install_checkpoints() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (url, expected_sha) in CHECKPOINTS.items():
        destination = CHECKPOINT_DIR / name
        if destination.is_file() and sha256(destination) == expected_sha:
            print(f"{name} ya está descargado y verificado.")
            continue
        temporary = destination.with_suffix(destination.suffix + ".download")
        temporary.unlink(missing_ok=True)
        run([sys.executable, "-m", "gdown", url, "-O", str(temporary)])
        actual_sha = sha256(temporary)
        if actual_sha != expected_sha:
            temporary.unlink(missing_ok=True)
            raise SystemExit(
                f"Checkpoint {name} inválido: esperado {expected_sha}, obtenido {actual_sha}."
            )
        temporary.replace(destination)
        print(f"Checkpoint {name} verificado: {expected_sha}")


def main() -> None:
    install_source()
    install_checkpoints()
    print("VM-UNet ISIC 2017 e ISIC 2018 están listos para la ruta CPU de referencia.")


if __name__ == "__main__":
    main()
