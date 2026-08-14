"""Small, file-backed contracts shared by interruptible HPC stages."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import stat
from typing import Any

from .benchmark import atomic_write_json, sha256_file


PHASE_STATUSES = {"running", "interrupted", "failed", "completed"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def artifact_identity(path: Path) -> dict[str, Any]:
    """Return a strict identity for a regular, non-empty required artifact."""
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        raise ValueError(f"required artifact is not a non-empty regular file: {path}")
    return {"path": str(path.resolve()), "bytes": info.st_size, "sha256": sha256_file(path)}


def write_phase_state(
    path: Path,
    *,
    phase: str,
    status: str,
    contract: dict[str, Any],
    previous: dict[str, Any] | None = None,
    **details: Any,
) -> dict[str, Any]:
    """Atomically persist one transition; callers retain their domain manifest."""
    if status not in PHASE_STATUSES:
        raise ValueError(f"invalid HPC phase status: {status}")
    state = {
        "schema_version": 1,
        "phase": phase,
        "status": status,
        "updated_utc": utc_now(),
        "contract": contract,
        **details,
    }
    if previous:
        state["started_utc"] = previous.get("started_utc", state["updated_utc"])
        state["attempt"] = int(previous.get("attempt", 0)) + (status == "running")
    else:
        state["started_utc"] = state["updated_utc"]
        state["attempt"] = 1 if status == "running" else 0
    atomic_write_json(path, state)
    return state
