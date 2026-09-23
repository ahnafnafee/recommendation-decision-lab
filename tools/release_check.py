"""Refuse files outside the code-and-aggregate public release boundary."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent.parent
ROOT_FILES = {".gitignore", ".gitattributes", "README.md", "DATA_LICENSE.md", "LICENSE",
              "CITATION.cff", "pyproject.toml"}
EXACT = {"web/index.html", "web/app.js", "web/demo_bundle.json",
         ".github/workflows/verify.yml", ".github/workflows/pages.yml",
         "tools/release_check.py"}
FORBIDDEN_JSON_KEYS = {"user_id", "parent_asin", "history", "target", "ranked", "weights", "model"}


def allowed(relative: str) -> bool:
    parts = Path(relative).parts
    if relative in ROOT_FILES or relative in EXACT:
        return True
    return (len(parts) == 2 and
            ((parts[0] in {"reliability", "tests"} and parts[1].endswith(".py")) or
             (parts[0] == "protocol" and parts[1].endswith(".md")) or
             (parts[0] == "reports" and parts[1].endswith((".md", ".json")))))


def scan_aggregate(value):
    if isinstance(value, dict):
        if FORBIDDEN_JSON_KEYS.intersection(value):
            raise ValueError("aggregate contains row or model fields")
        for child in value.values():
            scan_aggregate(child)
    elif isinstance(value, list):
        if len(value) > 100:
            raise ValueError("aggregate contains a suspiciously long array")
        for child in value:
            scan_aggregate(child)


def main():
    output = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                                     cwd=ROOT, text=True)
    paths = sorted(set(output.splitlines()))
    if not paths:
        raise ValueError("empty release")
    for relative in paths:
        path = ROOT / relative
        if not allowed(relative) or not path.is_file() or path.is_symlink():
            raise ValueError(f"file outside public allowlist: {relative}")
        if relative.startswith("reports/") and relative.endswith(".json"):
            if path.stat().st_size > 200_000:
                raise ValueError(f"aggregate too large: {relative}")
            scan_aggregate(json.loads(path.read_text(encoding="utf-8")))
    print(f"release boundary passed: {len(paths)} source, documentation and aggregate files")


if __name__ == "__main__":
    main()
