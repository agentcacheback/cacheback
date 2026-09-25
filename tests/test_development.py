"""Exercise both agent-event shapes and the shared enforcement paths."""

import importlib.util
import io
import json
import sys
from pathlib import Path
from typing import Any


def test_harness_catches_bypasses_comments_and_both_agent_edit_formats(
    tmp_path: Path, monkeypatch: Any
) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    import agent_hook
    import block_no_verify
    import check

    for command in (
        "git commit --no-verify",
        "git commit -nm message",
        "git push --no-verify",
        "git status;git commit -n",
        "git status\ngit commit -n",
        "git -c core.hooksPath=/tmp commit",
        "git config --unset core.hooksPath",
        "rm scripts/check.py",
        "mv .codex/hooks.json /tmp/hooks.json",
    ):
        assert block_no_verify.blocked_reason(command), command
    assert block_no_verify.blocked_reason('git commit -m "mention --no-verify"') is None
    target = tmp_path / "edited.py"
    target.write_text("# first\n# second\n# third\nx = 1\n")
    assert check.text_checks([str(target)]) == 1
    target.write_text("# first\n# second\nx = 1\n")
    assert check.text_checks([str(target)]) == 0
    checked = []
    monkeypatch.setattr(agent_hook, "ROOT", tmp_path)
    monkeypatch.setattr(agent_hook, "text_checks", lambda paths: checked.extend(paths) or 0)
    for details in (
        {"file_path": str(target)},
        {"command": "*** Begin Patch\n*** Update File: edited.py\n*** End Patch"},
        {"command": "*** Begin Patch\n*** Move to: edited.py\n*** End Patch"},
    ):
        monkeypatch.setattr(
            sys, "stdin", io.StringIO(json.dumps({"cwd": str(tmp_path), "tool_input": details}))
        )
        assert agent_hook.main() == 0
    assert checked == [str(target)] * 3
    assert importlib.util.find_spec("check_file_size") is not None
