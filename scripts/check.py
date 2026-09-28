"""The main-branch harness: shared text gates, lint, strict types, complexity and CPU tests."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tokenize
from pathlib import Path

from check_file_size import main as check_sizes
from check_secrets import main as check_secrets

ROOT = Path(__file__).resolve().parent.parent
TEXT = {".py", ".md", ".sh", ".toml", ".yaml", ".yml", ".json", ".txt", ".cff"}
# Counts inherited from the library at complexity thresholds 10, 15, 20 and 25; only shrink.
COMPLEXITY = {10: 4, 15: 1, 20: 0, 25: 0}


def text_checks(paths: list[str]) -> int:
    """Apply the same text checks to agent edits, local hooks and CI."""
    failed = check_sizes(paths) | check_secrets(paths)
    for name in paths:
        path = Path(name)
        if not path.is_file() or path.suffix not in TEXT:
            continue
        source = path.read_text(encoding="utf-8")
        if any(
            chr(code) in source for code in (0x2013, 0x2014, 0x200B, 0x200C, 0x200D, 0xFEFF, 0x2060)
        ):
            print(f"BLOCK {name}: punctuation dash or invisible character")
            failed = 1
        if path.suffix != ".py":
            continue
        previous, count = -1, 0
        try:
            lines = source.splitlines()
            for token in tokenize.generate_tokens(io.StringIO(source).readline):
                if (
                    token.type != tokenize.COMMENT
                    or lines[token.start[0] - 1][: token.start[1]].strip()
                ):
                    continue
                count = count + 1 if token.start[0] == previous + 1 else 1
                previous = token.start[0]
                if count == 3:
                    print(f"BLOCK {name}:{previous - 2}: comments must be fewer than 3 lines")
                    failed = 1
        except tokenize.TokenError as error:
            print(f"BLOCK {name}: {error}")
            failed = 1
    return failed


def check_install() -> int:
    """Require all three Git hooks and both agent integrations."""
    configured = subprocess.run(
        ["git", "config", "core.hooksPath"], capture_output=True, text=True, check=False
    ).stdout.strip()
    if configured != ".githooks":
        print("BLOCK: run scripts/install-hooks.sh before developing")
        return 1
    for name in ("pre-commit", "pre-push", "pre-merge-commit"):
        hook = ROOT / ".githooks" / name
        if not hook.is_file() or not os.access(hook, os.X_OK):
            print(f"BLOCK: missing executable hook {name}")
            return 1
        if 'exec "$root/.venv/bin/python" "$root/scripts/check.py"' not in hook.read_text():
            print(f"BLOCK: {name} no longer runs the shared harness")
            return 1
    for config in (".codex/hooks.json", ".claude/settings.json"):
        hooks = json.loads((ROOT / config).read_text())["hooks"]
        for event, script in (
            ("PreToolUse", "block_no_verify.py"),
            ("PostToolUse", "agent_hook.py"),
        ):
            commands = [h["command"] for group in hooks[event] for h in group["hooks"]]
            if not any(f"scripts/{script}" in command for command in commands):
                print(f"BLOCK: {config} does not wire {script}")
                return 1
    return 0


def main() -> int:
    """Run the complete offline harness, or text checks for an edit's explicit paths."""
    os.chdir(ROOT)
    if sys.argv[1:2] == ["--edit"]:
        return text_checks(sys.argv[2:])
    failed = check_install()
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], text=True
    )
    failed |= text_checks(sorted(set(tracked.strip("\0").split("\0"))))
    for command in (
        [sys.executable, "-m", "ruff", "format", "--check", "."],
        [sys.executable, "-m", "ruff", "check", "."],
        [sys.executable, "-m", "pyright"],
    ):
        failed |= int(subprocess.run(command, check=False).returncode != 0)
    for threshold, ceiling in COMPLEXITY.items():
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "src",
                "--select",
                "C901",
                "--config",
                f"lint.mccabe.max-complexity={threshold}",
                "--output-format",
                "json",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if (
            result.returncode not in (0, 1)
            or result.stderr
            or len(json.loads(result.stdout)) > ceiling
        ):
            print(f"BLOCK: complexity count above {threshold} exceeds {ceiling}, or lint failed")
            failed = 1
    failed |= int(subprocess.run([sys.executable, "-m", "pytest"], check=False).returncode != 0)
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
