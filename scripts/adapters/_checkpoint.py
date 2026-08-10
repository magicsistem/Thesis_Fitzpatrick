"""Permit native fixed checkpoints or explicitly identified B2 checkpoints."""

from __future__ import annotations

import json
from pathlib import Path


def verify_checkpoint(path: Path, actual_sha256: str, native_sha256: str, backend_id: str) -> None:
    if actual_sha256 == native_sha256:
        return
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    if not metadata_path.is_file():
        raise ValueError(f"SHA-256 nativo inválido y falta identidad B2: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("protocol") != "B2" or metadata.get("backend_id") != backend_id or metadata.get("checkpoint_sha256") != actual_sha256:
        raise ValueError(f"El checkpoint no es nativo ni un B2 identificado para {backend_id}")
