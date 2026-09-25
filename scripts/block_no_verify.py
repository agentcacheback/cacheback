#!/usr/bin/env python3
"""Agent-session PreToolUse guard for common Git-hook bypass commands."""

from __future__ import annotations

import json
import shlex
import sys

#: Shell operators that end one command and start another inside a Bash string.
SEPARATORS = {"&&", "||", ";", "|", "&", "\n"}

#: git-commit options whose following token is a value, not a flag.
VALUE_OPTS = {"-m", "--message", "-F", "--file", "-C", "--reuse-message", "-t", "--template"}


def _skips_hooks(span: list[str]) -> bool:
    """Return True when one command span is a git commit with a no-verify flag."""
    if not span or (span[0] != "git" and not span[0].endswith("/git")):
        return False
    if "commit" not in span:
        return False
    args = span[span.index("commit") + 1 :]
    i = 0
    while i < len(args):
        tok = args[i]
        if tok == "--":
            break
        if tok == "--no-verify":
            return True
        if tok in VALUE_OPTS:
            i += 2
            continue
        if tok.startswith("-") and not tok.startswith("--") and len(tok) > 1:
            letters = tok[1:]
            if "n" in letters:
                return True
            if "m" in letters or "F" in letters:
                i += 2
                continue
        i += 1
    return False


#: Files an agent must never delete or move out from under the harness.
PROTECTED_PREFIXES = (
    ".githooks",
    ".codex",
    ".claude",
    "scripts/check.py",
    "scripts/check_",
    "scripts/agent_hook.py",
    "scripts/install-hooks.sh",
    "scripts/block_no_verify",
)

#: git subcommands where --no-verify skips a hook.
NO_VERIFY_CMDS = {"commit", "push", "merge", "rebase"}


def _is_git(tok: str) -> bool:
    return tok == "git" or tok.endswith("/git")


def _span_blocked(span: list[str]) -> str | None:
    """Return a reason string when one command span bypasses the harness."""
    if not span:
        return None
    head = span[0]
    if head in {"rm", "mv"} or head.endswith(("/rm", "/mv")):
        for tok in span[1:]:
            clean = tok[2:] if tok.startswith("./") else tok
            if clean.startswith(PROTECTED_PREFIXES):
                return f"deleting or moving harness file '{tok}'"
        return None
    # Skip env-var prefixes (VAR=value) to find the real command word.
    i = 0
    git_dir_env = False
    while i < len(span) and "=" in span[i] and not span[i].startswith("-"):
        if span[i].startswith("GIT_DIR="):
            git_dir_env = True
        i += 1
    if i >= len(span) or not _is_git(span[i]):
        return None
    args = span[i + 1 :]
    subcommand = next((a for a in args if not a.startswith("-")), "")
    for j, tok in enumerate(args):
        if tok == "-c" and j + 1 < len(args) and args[j + 1].startswith("core.hooksPath"):
            return "git -c core.hooksPath override"
        if tok.startswith("-ccore.hooksPath") or tok == "--git-dir" or tok.startswith("--git-dir="):
            return "git-dir/hooksPath override"
    if git_dir_env and subcommand in NO_VERIFY_CMDS:
        return "GIT_DIR override on a gated command"
    if subcommand == "config":
        rest = [a for a in args if a != "config"]
        for j, tok in enumerate(rest):
            if tok.endswith("core.hooksPath") or tok == "core.hooksPath":
                value = rest[j + 1] if j + 1 < len(rest) else ""
                if not value.endswith(".githooks"):
                    return "git config core.hooksPath change"
        if "--unset" in rest and any("core.hooksPath" in t for t in rest):
            return "git config core.hooksPath unset"
    if subcommand in NO_VERIFY_CMDS and subcommand != "commit" and "--no-verify" in args:
        return f"git {subcommand} --no-verify"
    if _skips_hooks(span[i:]):
        return "git commit --no-verify"
    return None


def blocked_reason(command: str) -> str | None:
    """Return the bypass reason when any command in the Bash string has one."""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|\n")
        lexer.whitespace = " \t\r"
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return None
    span: list[str] = []
    for tok in tokens:
        if tok in SEPARATORS:
            reason = _span_blocked(span)
            if reason:
                return reason
            span = []
        else:
            span.append(tok)
    return _span_blocked(span)


def main() -> int:
    """Read the hook payload from stdin and veto no-verify commits."""
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0
    command = payload.get("tool_input", {}).get(
        "command", payload.get("tool_input", {}).get("cmd", "")
    )
    # Codex sends the shell command as an argv list (often bash -lc <script>);
    # Claude sends a plain string. Normalize to the script text.
    if isinstance(command, list):
        parts = [str(part) for part in command]
        if len(parts) >= 3 and parts[0].rsplit("/", 1)[-1] in {"bash", "sh", "zsh"}:
            command = parts[2] if parts[1] in {"-lc", "-c", "-lic"} else " ".join(parts)
        else:
            command = " ".join(parts)
    reason = blocked_reason(command)
    if reason is not None:
        print(
            f"harness: blocked ({reason}). Hook bypasses are banned in this "
            "repo (AGENTS.md); fix the violations instead of skipping the gates.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
