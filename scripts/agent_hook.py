"""Feed Codex patch paths and Claude Edit/Write paths into the shared text gates."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from check import ROOT, text_checks


def main() -> int:
    """Read the post-edit event and check files inside this project."""
    event = json.load(sys.stdin)
    details = event.get("tool_input", {})
    paths = [details["file_path"]] if details.get("file_path") else []
    patch = details.get("command", "")
    if isinstance(patch, list):
        patch = "\n".join(patch)
    paths.extend(
        re.findall(r"^\*\*\* (?:(?:Add|Update) File:|Move to:) (.+)$", patch, re.MULTILINE)
    )
    base = Path(event.get("cwd", ROOT))
    resolved = [str((base / name).resolve()) for name in paths]
    inside = [name for name in resolved if Path(name).is_relative_to(ROOT)]
    return 2 if text_checks(inside) else 0


if __name__ == "__main__":
    raise SystemExit(main())
