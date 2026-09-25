"""The selector-development contract: budgets, the input importer, and the bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from rcc.data import BUNDLES, _register, sha256_file
from rcc.run.selector_dev.capture import selections
from rcc.run.selector_dev.contract import (
    BUNDLE_SHA256,
    CELLS,
    MODELS,
    RATIOS,
    SELECTORS,
    import_inputs,
    token_hash,
)

LENGTH = 46_666


def test_selections_hold_the_budget_and_geometry():
    scores = {name: torch.arange(LENGTH, dtype=torch.float32) for name in SELECTORS}
    rows = selections(scores)
    assert len(rows) == CELLS == 1 + len(SELECTORS) * len(RATIOS)
    assert rows[0] == {"selector": "uncompressed", "ratio": 1, "keep": list(range(LENGTH))}
    for row in rows[1:]:
        keep = row["keep"]
        assert row["ratio"] in RATIOS
        assert len(keep) == (LENGTH + row["ratio"] - 1) // row["ratio"]
        # The sink and every rolled latent row survive at every ratio.
        assert 0 in keep and set(range(LENGTH - 40, LENGTH)).issubset(keep)
        assert keep == sorted(set(keep))


def test_selections_refuse_a_short_or_partial_roster():
    scores = {name: torch.arange(LENGTH, dtype=torch.float32) for name in SELECTORS}
    with pytest.raises(ValueError):
        selections({name: score for name, score in scores.items() if name != "h2o"})
    with pytest.raises(ValueError):
        selections({name: torch.arange(8, dtype=torch.float32) for name in SELECTORS})


class _EchoTokenizer:
    """Re-encodes the audited ids; ``drift`` makes one call disagree instead."""

    def __init__(self, ids: list[int], *, drift: bool = False) -> None:
        self._ids = [*ids[:-1], ids[-1] + 1] if drift else ids

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": self._ids}


def _write_prepared(root: Path, qids: list[str], ids: list[int]) -> Path:
    """Write a prepared bank of thirty audited stand-in prompts."""
    prepared = root / "prepared"
    (prepared / "items").mkdir(parents=True)
    rows = []
    for qid in qids:
        item = {
            "qid": qid,
            "question": f"Where is {qid}?",
            "prompt_text": "rivers",
            "prompt_token_ids": ids,
            "prompt_sha256": token_hash(ids),
            "source_composition": {
                "padding_source": "none",
                "padding_payload_tokens": 0,
                "padding_correction_tokens": 0,
                "final_prompt_tokens": len(ids),
            },
        }
        path = prepared / "items" / f"{qid}.json"
        path.write_text(json.dumps(item), encoding="utf-8")
        rows.append({"qid": qid, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (prepared / "manifest.json").write_text(
        json.dumps({"qids": qids, "items": rows}), encoding="utf-8"
    )
    return prepared


@pytest.fixture
def prepared_bank(tmp_path, monkeypatch):
    """A synthetic thirty-question prepared bank and the freeze that names it."""
    import rcc.run.selector_dev.contract as contract

    ids = [7, 8, 9]
    qids = [f"q{index:02d}" for index in range(30)]
    prepared = _write_prepared(tmp_path, qids, ids)
    index = tmp_path / "fanout-final-dev.json"
    index.write_text("[]", encoding="utf-8")
    questions = [
        type("Q", (), {"qid": qid, "question": f"Where is {qid}?", "raw": {}})() for qid in qids
    ]
    monkeypatch.setattr(contract, "load_freeze", lambda: {"dev_ids": qids})
    monkeypatch.setattr(contract, "load_questions", lambda _path: questions)
    return prepared, index, ids


def _run_import(root: Path, prepared: Path, index: Path, tokenizers: list[Any], monkeypatch):
    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda _cls, *a, **k: tokenizers.pop(0)),
    )
    return import_inputs(root, prepared, index, commit="0" * 40)


def test_import_accepts_the_audited_bank(tmp_path, prepared_bank, monkeypatch):
    prepared, index, ids = prepared_bank
    tokenizers = [_EchoTokenizer(ids) for _ in MODELS]
    plan = _run_import(tmp_path / "run", prepared, index, tokenizers, monkeypatch)
    assert [row["qid"] for row in plan["items"]] == [f"q{index:02d}" for index in range(30)]
    assert plan["source_commit"] == "0" * 40


def test_import_refuses_a_roster_that_is_not_the_frozen_dev30(tmp_path, prepared_bank, monkeypatch):
    prepared, index, ids = prepared_bank
    manifest = json.loads((prepared / "manifest.json").read_text())
    manifest["qids"] = [*manifest["qids"][:-1], "not-a-dev-question"]
    (prepared / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    tokenizers = [_EchoTokenizer(ids) for _ in MODELS]
    with pytest.raises(ValueError, match="frozen dev30 roster"):
        _run_import(tmp_path / "run", prepared, index, tokenizers, monkeypatch)


def test_import_refuses_a_prompt_a_tokenizer_re_encodes_differently(
    tmp_path, prepared_bank, monkeypatch
):
    prepared, index, ids = prepared_bank
    tokenizers = [_EchoTokenizer(ids) for _ in MODELS]
    tokenizers[-1] = _EchoTokenizer(ids, drift=True)
    with pytest.raises(ValueError, match="tokenizer changes the prompt"):
        _run_import(tmp_path / "run", prepared, index, tokenizers, monkeypatch)


def test_the_shipped_capture_bundle_carries_its_pinned_digest():
    """The archive beside the tree, when present, is the one the constant names."""
    _register()
    digest, archive_name = BUNDLES["selector-dev-v8"]
    archive = Path(__file__).resolve().parents[1] / "data" / archive_name
    if not archive.is_file():
        pytest.skip("the selector-dev archive is not beside this tree")
    assert sha256_file(archive) == digest == BUNDLE_SHA256
