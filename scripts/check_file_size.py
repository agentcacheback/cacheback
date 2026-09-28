#!/usr/bin/env python3
"""File-size gate for the commit harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

WARN = 400
BLOCK = 600
MAX_LINE_CHARS = 3000

CHECK_SUFFIXES = {".py", ".md", ".sh", ".toml", ".yaml", ".yml", ".json"}
SKIP_PARTS = {".git", ".devnotes", "node_modules", ".venv", "venv", "__pycache__"}
EXEMPT_FILE = Path(__file__).resolve().parent / "size_exempt.txt"


def load_exemptions() -> tuple[set[str], list[str]]:
    """Return (exempt path strings, stale entries whose paths are gone)."""
    exempt: set[str] = set()
    stale: list[str] = []
    if not EXEMPT_FILE.is_file():
        return exempt, stale
    root = EXEMPT_FILE.parent.parent
    for raw in EXEMPT_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        rel = line.split(" : ", 1)[0].strip()
        if (root / rel).is_file():
            exempt.add(rel)
        else:
            stale.append(rel)
    return exempt, stale


def non_blank_lines(path: Path) -> int:
    """Count non-blank lines; unreadable files count as zero."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeError):
        return 0
    return sum(1 for line in text.splitlines() if line.strip())


def notebook_source_lines(path: Path) -> int:
    """Count non-blank cell-source lines; an unparseable notebook counts as zero."""
    try:
        notebook = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, UnicodeError, ValueError):
        return 0
    count = 0
    for cell in notebook.get("cells", []):
        source = cell.get("source", "")
        raw = source if isinstance(source, list) else str(source).splitlines()
        count += sum(1 for entry in raw if str(entry).strip())
    return count


def longest_line(path: Path) -> int:
    """Return the longest line length; unreadable files count as zero."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeError):
        return 0
    return max((len(line) for line in text.splitlines()), default=0)


def should_check(path: Path) -> bool:
    """Return True for source/docs files outside skipped trees."""
    if any(part in SKIP_PARTS for part in path.parts):
        return False
    return path.suffix in CHECK_SUFFIXES or path.suffix == ".ipynb"


def is_exempt(path: Path, exempt: set[str]) -> bool:
    """Match a possibly-absolute or temp-tree path against exempt entries."""
    posix = path.as_posix()
    return any(posix == rel or posix.endswith("/" + rel) for rel in exempt)


def main(argv: list[str]) -> int:
    """Gate the given files; exit 1 when any exceeds the block threshold."""
    exempt, stale = load_exemptions()
    blocked = False
    for rel in stale:
        print(f"BLOCK  scripts/size_exempt.txt: entry '{rel}' no longer exists; remove it.")
        blocked = True
    for arg in argv:
        path = Path(arg)
        if not path.is_file() or not should_check(path):
            continue
        if is_exempt(path, exempt):
            continue
        if path.suffix == ".ipynb":
            loc = notebook_source_lines(path)
            if loc > BLOCK:
                print(f"warn   {path}: {loc} cell-source lines (soft limit {BLOCK} for notebooks).")
            continue
        loc = non_blank_lines(path)
        if loc > BLOCK:
            print(f"BLOCK  {path}: {loc} lines (limit {BLOCK}). Split this file.")
            blocked = True
        elif loc > WARN:
            print(f"warn   {path}: {loc} lines (soft limit {WARN}). Consider splitting.")
        if path.suffix not in {".ipynb", ".md"}:
            width = longest_line(path)
            if width > MAX_LINE_CHARS:
                print(
                    f"BLOCK  {path}: a {width}-char line (limit {MAX_LINE_CHARS}); "
                    "looks like an embedded blob."
                )
                blocked = True
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
