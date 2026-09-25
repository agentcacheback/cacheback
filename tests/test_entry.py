"""One config, one bundle, one command: `rcc.run.entry` end to end on CPU.

Everything between the config file and `report.json` is shipped code. Faked:
the tokenizer download, the tokenizer, the engine, and threads for seats.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import transformers
from tests.longbench_coa_support import FakeTokenizer
from tests.natural_bundle import write_bundle

from rcc.benchmarks.fanoutqa import preflight, qwen_serving
from rcc.benchmarks.fanoutqa.qwen_serving import panel_policy_config, policy_config_fingerprint
from rcc.models.qwen import QWEN_FAMILY
from rcc.models.qwen.capture import selected_indices_sha256
from rcc.models.qwen.prompts import worker_prompts
from rcc.run import cli, entry, rescore, split_driver
from rcc.run import handoff_recall as recall
from rcc.run import plan as plan_module
from rcc.run.fleet.schedule import run_in_threads
from rcc.run.qwen import execution as qwen_execution
from rcc.run.qwen import worker as qwen_worker

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "qwen-fanoutqa-natural-dev50.toml"
_BENCHMARK_KEY = "fanoutqa-natural-dev50"
_VOCAB = 1024
_HIDDEN = 8


class _Tokenizer(FakeTokenizer):
    """The shared fake at one character per id, so a decode gives the text back."""

    def decode(self, tokens: Any, **kwargs: Any) -> str:
        special = {
            QWEN_FAMILY.think_open_token_ids[0]: QWEN_FAMILY.think_open,
            QWEN_FAMILY.think_close_token_ids[0]: QWEN_FAMILY.think_close,
            **dict.fromkeys(QWEN_FAMILY.stop_token_ids, ""),
        }
        return "".join(special.get(int(token), chr(int(token) - 1)) for token in tokens)


class _Engine:
    """Finish every admitted request on the next step with one closed answer."""

    def __init__(self, tokenizer: _Tokenizer) -> None:
        self.tokenizer = tokenizer
        self._live: dict[str, Any] = {}
        self.aborted: list[str] = []

    def sampling(self, sampling: Any, seed: int) -> Any:
        return SimpleNamespace(seed=seed, max_tokens=sampling.max_tokens)

    def add_request(self, request_id: str, prompt: Any, sampling: Any) -> None:
        self._live[request_id] = (prompt, sampling)

    def _ids(self) -> list[int]:
        return [
            QWEN_FAMILY.think_open_token_ids[0],
            *self.tokenizer.encode("thinking"),
            QWEN_FAMILY.think_close_token_ids[0],
            *self.tokenizer.encode("Alpha and Beta"),
            QWEN_FAMILY.stop_token_ids[0],
        ]

    def decode_text_full(self, prompts: list[str], requests: list[Any]) -> list[Any]:
        ids = self._ids()
        return [
            SimpleNamespace(
                request_id=request.request_id,
                prompt_token_ids=tuple(self.tokenizer.encode(prompt)),
                text=self.tokenizer.decode(ids),
                n_tokens=len(ids),
                token_ids=tuple(ids),
                finish_reason="stop",
                num_cached_tokens=0,
                queued_ts=None,
                scheduled_ts=None,
                first_token_ts=None,
            )
            for prompt, request in zip(prompts, requests, strict=True)
        ]

    def step(self) -> list[Any]:
        ids = self._ids()
        outputs = [
            SimpleNamespace(
                request_id=request_id,
                finished=True,
                num_cached_tokens=0,
                metrics=None,
                outputs=[
                    SimpleNamespace(
                        token_ids=list(ids),
                        text=self.tokenizer.decode(ids),
                        finish_reason="stop",
                    )
                ],
            )
            for request_id in self._live
        ]
        self._live.clear()
        return outputs

    def has_unfinished_requests(self) -> bool:
        return bool(self._live)

    def abort_request(self, request_ids: list[str]) -> None:
        self.aborted.extend(request_ids)


def _backend(tokenizer: _Tokenizer) -> Any:
    """The resident receiver: a warm-up, an embedding table, a KV pool, an engine."""
    table = torch.arange(_VOCAB * _HIDDEN, dtype=torch.float32).reshape(_VOCAB, _HIDDEN)
    engine = _Engine(tokenizer)
    return SimpleNamespace(
        engine=engine,
        decode_text_full=engine.decode_text_full,
        warmup_decode=lambda _prompt, *, seed: None,
        resident_model=lambda: SimpleNamespace(
            get_input_embeddings=lambda: SimpleNamespace(weight=table)
        ),
        kv_pool_tokens=lambda: 1_000_000,
        close=lambda: None,
    )


@pytest.fixture
def cpu_node(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stand one node up on CPU: the bundle, the profile, the fakes, the config."""
    bundle = tmp_path / "bundle"
    profile = dataclasses.replace(
        write_bundle(bundle),
        worker_prompt_tokens=2_000,
        # The rescore reads the prepared panel back at the profile's own source
        # commit, which for a shipped profile is the commit its panel was sealed
        # at; here that is the commit the entry seals it at.
        source_commit=plan_module._git_commit(),
    )
    # The sealed policy-config fingerprint covers the item count and the prompt
    # width, so the stand-in re-pins it from live code the way every pin is.
    profile = dataclasses.replace(
        profile,
        prepared_config_fingerprint=policy_config_fingerprint(
            panel_policy_config(profile, family=QWEN_FAMILY)
        ),
    )
    tokenizer = _Tokenizer(width=1)
    # One id per character spends four times the real rate, so the manager
    # ceiling moves with it in both places that read it.
    for module in (preflight, qwen_serving):
        monkeypatch.setattr(module, "MANAGER_PROMPT_CEILING", 4 * module.MANAGER_PROMPT_CEILING)
    monkeypatch.setitem(plan_module._BENCHMARKS, _BENCHMARK_KEY, profile)
    prefetched: list[tuple[str, str]] = []
    monkeypatch.setattr(
        entry, "prefetch_checkpoints", lambda pins, **_kwargs: prefetched.extend(pins)
    )
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_a, **_k: tokenizer)
    monkeypatch.setattr(qwen_worker, "build_qwen_backend", lambda *_a, **_k: _backend(tokenizer))
    monkeypatch.setattr(qwen_worker, "VllmEngineHandle", lambda backend, **_k: backend.engine)
    monkeypatch.setattr(split_driver, "run_in_processes", run_in_threads)
    saved = dict(os.environ)
    yield_value = {
        "bundle": bundle,
        "profile": profile,
        "tokenizer": tokenizer,
        "prefetched": prefetched,
    }
    config = tmp_path / "qwen-one-item.toml"
    text = _CONFIG.read_text(encoding="utf-8")
    text = text.replace("count = 50", "count = 1").replace(
        'root = "runs"', f'root = "{tmp_path / "runs"}"'
    )
    config.write_text(text, encoding="utf-8")
    yield_value["config"] = config
    try:
        yield yield_value
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_the_entry_runs_one_config_to_a_rescored_report(
    cpu_node: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arm = QWEN_FAMILY.policy("issue_only")
    argv = [
        "--config",
        str(cpu_node["config"]),
        "--bundle",
        str(cpu_node["bundle"]),
        "--arm",
        arm,
        "--attempt-id",
        "a1",
    ]
    assert entry.main(argv) == 0
    # The prefetch names the receiver and every registered text sender, once each.
    profile = QWEN_FAMILY.profile
    senders = {
        (arm.sender_tokenizer, arm.sender_tokenizer_revision)
        for arm in profile.physical_arms
        if arm.sender_tokenizer is not None
    }
    prefetched = cpu_node["prefetched"]
    assert len(prefetched) == len(set(prefetched))
    assert set(prefetched) == {(profile.tokenizer, profile.tokenizer_revision), *senders}
    # The split driver prints its run and report payloads first; the report is last.
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    plan = plan_module.load_and_resolve(cpu_node["config"])
    root = Path(plan.output_uri)
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    assert printed == report
    assert report["qids"] == ["q1"] and report["policies"] == [arm]
    assert report["result_rows"] == 1 and report["paired_comparisons"] == []
    assert report["policy_grid"][0]["n"] == 1
    assert report["generation_censoring"]["status"] == "complete_within_caps"
    assert (root / "control" / "pipeline_identity.json").is_file()
    assert (root / "control" / "placements.json").is_file()
    banks = sorted((root / "arms" / arm / "workers").glob("gpu*/raw.jsonl"))
    assert len(banks) == 8
    rows = [json.loads(line) for bank in banks for line in bank.read_text().splitlines()]
    results = [row for row in rows if row.get("kind") == "result"]
    assert len(results) == 1 and results[0]["qid"] == "q1" and results[0]["policy"] == arm
    assert len(results[0]["answers"]) == 3
    selected = root / "report" / "selected_results.jsonl"
    assert json.loads(selected.read_text().splitlines()[0])["qid"] == "q1"

    # The offline rescore reads the published rows back through the family's
    # own reader. The one row passes, then the command refuses: the audit is
    # over the whole grid, which this run cut to one of its eleven arms.
    argv = [
        "rescore-results",
        "--config",
        str(cpu_node["config"]),
        "--plan-out",
        str(root / "plan.json"),
        "--results",
        str(selected),
        "--runtime-bank",
        str(root),
        "--rescore-out",
        str(root / "rescore.json"),
    ]
    with pytest.raises(RuntimeError, match=f"{len(plan.arms) - 1} cells missing"):
        cli.main(argv)
    assert json.loads((root / "plan.json").read_text())["run_id"] == plan.run_id

    # The judger ablation rides the hop the benchmark name rides: a registered
    # run publishes no word, and a plan that names one publishes it for every
    # seat the driver spawns, where the execution profile applies it.
    assert "RCC_FANOUT_JUDGER_QUESTION" not in os.environ
    judger = "Why was Le Chaton Fat denied a small-business loan?"
    ablation = dataclasses.replace(
        plan, benchmark=dataclasses.replace(plan.benchmark, judger_question=judger)
    )
    entry._fanout_environment(ablation, "qwen")
    assert os.environ["RCC_FANOUT_JUDGER_QUESTION"] == ablation.benchmark.judger_question
    assert qwen_execution.execution_profile().judger_question == ablation.benchmark.judger_question
    entry._fanout_environment(plan, "qwen")
    assert "RCC_FANOUT_JUDGER_QUESTION" not in os.environ
    assert qwen_execution.execution_profile().judger_question is None
    monkeypatch.setenv("RCC_FANOUT_JUDGER_QUESTION", " ")
    with pytest.raises(ValueError, match="must carry text"):
        qwen_execution.execution_profile()


def test_retained_runs_never_join_across_a_dropped_position() -> None:
    words = ("New", "York", "unrelated", "York", "1,000.0")

    def decode(ids: Any) -> str:
        return " ".join(words[i] for i in ids)

    groups = (("New York",), ("1000",))
    retained = recall.retained_text(range(5), [0, 1, 4], decode)
    assert recall.coverage(groups, retained) == (True, True)
    # Both words survive, but not adjacently: the boundary keeps them apart.
    assert recall.coverage(groups, recall.retained_text(range(5), [0, 3], decode)) == (False, False)
    assert recall.coverage((("80",),), "the 180 mark") == (False,)
    # Latent rows sit past the prompt and carry no token.
    assert recall.retained_text(range(5), [2, 5, 6], decode) == "unrelated"
    for keep in ([2, 1], [-1, 0]):
        with pytest.raises(ValueError):
            recall.retained_text(range(5), keep, decode)


def test_the_readers_score_the_banked_handoffs_and_answers(
    cpu_node: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    text_arm = QWEN_FAMILY.policy("text_primary")
    argv = ["--config", str(cpu_node["config"]), "--bundle", str(cpu_node["bundle"])]
    assert entry.main([*argv, "--arm", text_arm, "--attempt-id", "a1"]) == 0
    plan = plan_module.load_and_resolve(cpu_node["config"])
    root = Path(plan.output_uri)

    # A latent row banks its keeps, not its text: worker 0 keeps the run that
    # spells the first river and one position past its prompt, worker 1 and 2
    # keep nothing.
    (item,), manifest = rescore.route_prepared_items(plan, root)
    tokenizer = cpu_node["tokenizer"]
    texts = worker_prompts(item, tokenizer, family=QWEN_FAMILY, profile=plan.benchmark)
    start = texts[0].index("Alpha")
    keeps = [[*range(start, start + 5), len(texts[0]) + 3], [], []]
    latent_arm = "latent_query_support_r4"
    row = {
        "kind": "result",
        "qid": "q1",
        "arm": QWEN_FAMILY.policy(latent_arm),
        "semantic_arm": latent_arm,
        "prepared_sha256": manifest["artifact_sha256"],
        "worker_prompt_tokens": [len(t) for t in texts],
        "keeps_by_worker": keeps,
        "selected_indices_sha256": selected_indices_sha256(keeps),
        "worker_prompt_sha256": [hashlib.sha256(t.encode()).hexdigest() for t in texts],
    }
    bank = root / "arms" / row["arm"] / "workers" / "gpu0" / "raw.jsonl"
    bank.parent.mkdir(parents=True)
    bank.write_text(json.dumps(row) + "\n", encoding="utf-8")

    common = ["--config", str(cpu_node["config"]), "--plan-out", str(root / "plan.json")]
    out = root / "recall.json"
    argv = ["handoff-recall", *common, "--run-root", str(root)]
    assert cli.main([*argv, "--source-bundle", str(cpu_node["bundle"]), "--out", str(out)]) == 0
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed == json.loads(out.read_text(encoding="utf-8"))
    by_arm = {summary["arm"]: summary for summary in printed["arms"]}
    assert printed["source"]["hits"] == 2 and printed["source"]["leaves"] == 2
    assert by_arm["text_primary"]["hits"] == 2
    assert by_arm[latent_arm] == {
        "arm": latent_arm,
        "questions": 1,
        "hits": 1,
        "leaves": 2,
        "micro_recall_percent": 50.0,
        "macro_recall_percent": 50.0,
    }
    row["keeps_by_worker"][0].pop(0)
    bank.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="banked selection hash"):
        cli.main([*argv, "--source-bundle", str(cpu_node["bundle"]), "--out", str(out)])

    scores = root / "scores.json"
    argv = ["score-results", *common, "--results", str(root / "report" / "selected_results.jsonl")]
    argv += ["--source-bundle", str(cpu_node["bundle"]), "--out", str(scores)]
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    (banked,) = report["policy_grid"]
    assert cli.main(argv) == 0
    body = json.loads(scores.read_text(encoding="utf-8"))
    assert body["references"] == "fanoutqa-equivalence-v1"
    assert body["arms"] == {
        "text_primary": {"items": 1, "loose": banked["loose"], "strict": banked["strict"]}
    }
    assert cli.main([*argv, "--references", "original"]) == 0
    body = json.loads(scores.read_text(encoding="utf-8"))
    assert body["references"] == "original"
    assert body["arms"] == {"text_primary": {"items": 1, "loose": 1.0, "strict": 1.0}}
