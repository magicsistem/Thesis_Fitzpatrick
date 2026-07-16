"""Resolve project paths without creating directories beside the repository."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_repo_path(value: str | Path) -> Path:
    """Resolve a path relative to the repository root, never to the shell CWD."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve(strict=False)


def require_within(path: Path, roots: tuple[Path, ...], label: str) -> Path:
    """Reject a path outside all approved roots before any directory is created."""
    resolved = path.resolve(strict=False)
    approved = tuple(root.resolve(strict=False) for root in roots)
    if not any(resolved == root or resolved.is_relative_to(root) for root in approved):
        allowed = ", ".join(str(root) for root in approved)
        raise SystemExit(f"{label} must be inside one of: {allowed}. Received: {resolved}")
    return resolved


def require_descendant(path: Path, roots: tuple[Path, ...], label: str) -> Path:
    """Like require_within, but reject the protected root directory itself."""
    resolved = require_within(path, roots, label)
    approved = tuple(root.resolve(strict=False) for root in roots)
    if resolved in approved:
        allowed = ", ".join(f"{root}/<name>" for root in approved)
        raise SystemExit(f"{label} must name a subdirectory such as: {allowed}")
    return resolved


def resolve_input(value: str | Path) -> Path:
    return resolve_repo_path(value)


def resolve_data_output(value: str | Path) -> Path:
    path = resolve_repo_path(value)
    return require_descendant(
        path,
        (REPO_ROOT / "data" / "raw", REPO_ROOT / "data" / "interim", REPO_ROOT / "data" / "processed"),
        "Data output",
    )


def resolve_report_output(value: str | Path) -> Path:
    path = resolve_repo_path(value)
    return require_descendant(path, (REPO_ROOT / "reports" / "tables",), "Report output")


def resolve_result_output(value: str | Path) -> Path:
    path = resolve_repo_path(value)
    return require_descendant(path, (REPO_ROOT / "results",), "Inference output")
