#!/usr/bin/env python3
"""Secret-pattern gate for the commit harness."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ALLOW_PRAGMA = "secret-scan: allow"

CHECK_SUFFIXES = {".py", ".md", ".sh", ".toml", ".yaml", ".yml", ".json", ".txt", ".cfg", ".conf"}
SKIP_PARTS = {".git", ".devnotes", "node_modules", ".venv", "venv", "__pycache__"}

PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("AWS access key id", re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b")),
    (
        "AWS secret key assignment",
        re.compile(r"aws_secret_access_key\s*[=:]\s*['\"]?[A-Za-z0-9/+=]{40}", re.IGNORECASE),
    ),
    ("GitHub token", re.compile(r"\b(ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,})\b")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{34,}\b")),
    ("Anthropic/OpenAI style key", re.compile(r"\bsk-(ant-)?[A-Za-z0-9_-]{32,}\b")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{20,}\.eyJ[A-Za-z0-9_-]{20,}")),
    ("basic-auth URL", re.compile(r"https?://[^/\s:@]+:[^/\s@]+@")),
    (
        "base64 next to AWS_ (credential embedding stays in aws_lib.sh)",
        re.compile(r"base64.{0,80}AWS_(SECRET|ACCESS)|AWS_(SECRET|ACCESS).{0,80}base64"),
    ),
)


def is_env_named(path: Path) -> bool:
    """True for .env-style filenames outside .pins/."""
    if ".pins" in path.parts:
        return False
    name = path.name
    return name == ".env" or name.endswith(".env")


def scan_line(line: str) -> list[str]:
    """Return the labels of every secret pattern present in the line."""
    if ALLOW_PRAGMA in line:
        return []
    return [label for label, pattern in PATTERNS if pattern.search(line)]


def iter_lines(path: Path):
    """Yield (location, line) pairs for a text file or notebook cell sources."""
    if path.suffix == ".ipynb":
        try:
            notebook = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, UnicodeError, ValueError):
            return
        for i, cell in enumerate(notebook.get("cells", [])):
            source = cell.get("source", "")
            raw = source if isinstance(source, list) else str(source).splitlines()
            for n, entry in enumerate(raw, 1):
                yield f"cell {i} line {n}", str(entry)
        return
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeError):
        return
    for n, line in enumerate(text.splitlines(), 1):
        yield str(n), line


def main(argv: list[str]) -> int:
    """Gate the given files; exit 1 on any secret-shaped content."""
    violations = 0
    for arg in argv:
        path = Path(arg)
        if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
            continue
        if is_env_named(path):
            print(f"BLOCK  {path}: env-named file must never be committed.")
            violations += 1
            continue
        if path.suffix not in CHECK_SUFFIXES and path.suffix != ".ipynb":
            continue
        # Skip the pattern-definition file itself, including a staged copy of the same path.
        if path.parts[-2:] == ("scripts", "check_secrets.py"):
            continue
        for where, line in iter_lines(path):
            for label in scan_line(line):
                print(f"BLOCK  {path}:{where}: {label}")
                violations += 1
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
