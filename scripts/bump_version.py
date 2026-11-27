#!/usr/bin/env python3
"""Synchronize version across AetioCore workspace, Python packages, R, docs, and citation metadata."""

import json
import re
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2 or not re.match(r"^\d+\.\d+\.\d+$", sys.argv[1]):
        sys.exit("Usage: python3 scripts/bump_version.py X.Y.Z")
    v = sys.argv[1]

    replacements = [
        ("Cargo.toml", r'^version = "\d+\.\d+\.\d+"', f'version = "{v}"'),
        ("semflow/Cargo.toml", r'^version = "\d+\.\d+\.\d+"', f'version = "{v}"'),
        ("semflow-dl/Cargo.toml", r'^version = "\d+\.\d+\.\d+"', f'version = "{v}"'),
        ("semflow/pyproject.toml", r'^version = "\d+\.\d+\.\d+"', f'version = "{v}"'),
        ("semflow-dl/pyproject.toml", r'^version = "\d+\.\d+\.\d+"', f'version = "{v}"'),
        ("semflow/python/aetiocore_sem/__init__.py", r'__version__ = "\d+\.\d+\.\d+"', f'__version__ = "{v}"'),
        ("semflow-dl/python/aetiocore_sem_dl/__init__.py", r'__version__ = "\d+\.\d+\.\d+"', f'__version__ = "{v}"'),
        ("bindings/r/DESCRIPTION", r'^Version: \d+\.\d+\.\d+', f'Version: {v}'),
        ("CITATION.cff", r'^version: "\d+\.\d+\.\d+"', f'version: "{v}"'),
        ("docs/index.md", r'v\d+\.\d+\.\d+', f'v{v}'),
        ("docs/index.md", r'\*\*Version:\*\* `\d+\.\d+\.\d+`', f'**Version:** `{v}`'),
    ]
    for path_str, pattern, repl in replacements:
        p = Path(path_str)
        if p.exists():
            text = p.read_text(encoding="utf-8")
            updated = re.sub(pattern, repl, text, count=1, flags=re.MULTILINE)
            p.write_text(updated, encoding="utf-8")

    zenodo = Path(".zenodo.json")
    if zenodo.exists():
        data = json.loads(zenodo.read_text(encoding="utf-8"))
        data["version"] = v
        zenodo.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    print(f"Synced AetioCore version to {v}")


if __name__ == "__main__":
    main()
