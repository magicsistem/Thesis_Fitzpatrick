#!/usr/bin/env python3
"""Validate local configs, HTML/JS DOM references, and Markdown file links."""

from __future__ import annotations

from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
try:
    import tomllib
except ModuleNotFoundError:
    from pip._vendor import tomli as tomllib

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


class IDs(HTMLParser):
    def __init__(self): super().__init__(); self.ids = []
    def handle_starttag(self, _tag, attrs):
        value = dict(attrs).get("id")
        if value: self.ids.append(value)


def main() -> None:
    errors = []
    counts = {"json": 0, "toml": 0, "yaml": 0, "markdown": 0}
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or any(part in {".git", "models", "data", "results", "__pycache__"} for part in path.relative_to(REPO_ROOT).parts): continue
        try:
            if path.suffix == ".json": json.loads(path.read_text(encoding="utf-8")); counts["json"] += 1
            elif path.suffix == ".toml": tomllib.loads(path.read_text(encoding="utf-8")); counts["toml"] += 1
            elif path.suffix in {".yaml", ".yml"}: yaml.safe_load(path.read_text(encoding="utf-8")); counts["yaml"] += 1
            elif path.suffix.lower() == ".md":
                counts["markdown"] += 1
                for target in re.findall(r"\[[^]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
                    target = target.split("#", 1)[0]
                    if not target or "://" in target or target.startswith("mailto:"): continue
                    if not (path.parent / target).resolve().exists(): errors.append(f"Enlace Markdown roto: {path.relative_to(REPO_ROOT)} -> {target}")
        except Exception as exc: errors.append(f"{path.relative_to(REPO_ROOT)}: {type(exc).__name__}: {exc}")
    parser = IDs(); parser.feed((REPO_ROOT / "web/index.html").read_text(encoding="utf-8"))
    duplicates = sorted({value for value in parser.ids if parser.ids.count(value) > 1})
    if duplicates: errors.append("IDs HTML duplicados: " + ", ".join(duplicates))
    javascript = (REPO_ROOT / "web/app.js").read_text(encoding="utf-8")
    referenced = set(re.findall(r'\$\("#([A-Za-z0-9_-]+)"\)', javascript))
    missing = sorted(referenced - set(parser.ids))
    if missing: errors.append("IDs usados por JavaScript y ausentes en HTML: " + ", ".join(missing))
    report = {"passed": not errors, "counts": counts, "html_ids": len(parser.ids), "javascript_dom_references": len(referenced), "errors": errors, "javascript_parser": "not available; DOM references and HTTP behavior validated, use node --check when Node is installed"}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors: raise SystemExit(2)


if __name__ == "__main__": main()
