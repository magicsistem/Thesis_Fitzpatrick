"""Verify the exact AViT inference runtime used by the CEDIA smoke job."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "models" / "avit" / "source"
REQUIRED_MODULES = ("numpy", "PIL", "torch", "torchvision", "timm", "einops")
HEADLESS_FILES = (
    SOURCE_DIR / "Models" / "CNN" / "ResNet.py",
    SOURCE_DIR / "Models" / "Decoders.py",
)


def main() -> None:
    import_errors: list[str] = []
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:
            import_errors.append(f"{name}: {type(exc).__name__}: {exc}")
    if import_errors:
        raise SystemExit(
            "Dependencias de AViT ausentes o incompatibles:\n- "
            + "\n- ".join(import_errors)
        )

    for path in HEADLESS_FILES:
        if not path.is_file():
            raise SystemExit(f"Fuente AViT ausente: {path}")
        if "from turtle import forward" in path.read_text(encoding="utf-8"):
            raise SystemExit(f"AViT conserva una importación gráfica no válida en HPC: {path}")

    sys.path.insert(0, str(SOURCE_DIR))
    try:
        from Models.Transformer.ViT_adapters import ViTSeg_CNNprompt_adapt  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            f"La ruta real de inferencia AViT no se puede importar: {type(exc).__name__}: {exc}"
        ) from exc
    print("AViT runtime verified: dependencies, headless patch and model import passed.")


if __name__ == "__main__":
    main()
