#!/usr/bin/env python3
"""Clone pinned public sources and verify every declared external resource."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "configs" / "hpc" / "external_resources.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clone_source(item: dict, execute: bool) -> dict:
    destination = REPO_ROOT / item["destination"]
    if not destination.exists():
        if not execute:
            return {"id": item["id"], "state": "would_clone", "destination": item["destination"]}
        staging = destination.with_name(destination.name + ".partial")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if staging.exists() and not (staging / ".git").is_dir():
            raise SystemExit(f"Incomplete non-Git source staging requires inspection: {staging}")
        if not staging.exists():
            subprocess.run(["git", "clone", "--no-checkout", item["repository"], str(staging)], check=True)
        present = subprocess.run(
            ["git", "cat-file", "-e", f"{item['revision']}^{{commit}}"], cwd=staging, check=False
        ).returncode == 0
        if not present:
            subprocess.run(["git", "fetch", "origin", item["revision"]], cwd=staging, check=True)
        subprocess.run(["git", "checkout", "--detach", item["revision"]], cwd=staging, check=True)
        staging.replace(destination)
    if not (destination / ".git").is_dir():
        raise SystemExit(f"Existing source is not a Git checkout: {destination}")
    present = subprocess.run(
        ["git", "cat-file", "-e", f"{item['revision']}^{{commit}}"], cwd=destination, check=False
    ).returncode == 0
    if not present:
        if not execute:
            return {"id": item["id"], "state": "would_fetch_revision", "revision": item["revision"]}
        subprocess.run(["git", "fetch", "origin", item["revision"]], cwd=destination, check=True)
    if execute:
        subprocess.run(["git", "checkout", "--detach", item["revision"]], cwd=destination, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=destination, check=True, capture_output=True, text=True).stdout.strip()
    if execute and head != item["revision"]:
        raise SystemExit(f"Pinned revision mismatch for {item['id']}: {head}")
    return {"id": item["id"], "state": "verified" if head == item["revision"] else "different_revision", "revision": head}


def verify_checkpoint(item: dict) -> dict:
    path = REPO_ROOT / item["destination"]
    if not path.is_file():
        return {"id": item["id"], "state": "missing", "destination": item["destination"]}
    actual_size = path.stat().st_size
    actual_sha = sha256(path)
    valid = actual_size == item["expected_bytes"] and actual_sha == item["sha256"]
    return {"id": item["id"], "state": "verified" if valid else "invalid", "bytes": actual_size, "sha256": actual_sha}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clone-sources", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Perform network/Git changes; otherwise report the plan")
    parser.add_argument("--require-checkpoints", action="store_true")
    parser.add_argument("--require-sources", action="store_true")
    args = parser.parse_args()
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sources = [clone_source(item, args.execute) for item in payload["sources"]] if args.clone_sources or args.require_sources else []
    checkpoints = [verify_checkpoint(item) for item in payload["checkpoints"]]
    report = {"schema_version": 1, "executed": args.execute, "expected_checkpoint_bytes": sum(item["expected_bytes"] for item in payload["checkpoints"]), "sources": sources, "checkpoints": checkpoints}
    print(json.dumps(report, indent=2))
    if args.require_checkpoints and any(item["state"] != "verified" for item in checkpoints):
        raise SystemExit(2)
    if args.require_sources and any(item["state"] != "verified" for item in sources):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
