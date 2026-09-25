"""The FanOutQA benchmark: its loader, panel, scorer, bundle, topology, and plan identities."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import io
import json
import pickle
from pathlib import Path

import pytest
from tests.natural_bundle import CharTokenizer, write_bundle

from rcc.benchmarks.fanoutqa import (
    FANOUTQA_NATURAL_DEV50,
    FANOUTQA_NATURAL_DEV50_LATENT,
    build_cli,
    scoring_equivalence,
)
from rcc.benchmarks.fanoutqa import prepare as prepare_module
from rcc.benchmarks.fanoutqa.data import (
    FANOUTQA_DATASET_REVISION,
    FANOUTQA_DEV_URL,
    Question,
    ShardPlan,
    evidence_answerability_audit,
    gold_leaves,
    load_freeze,
    real_fanout_structure_audit,
    score_text,
)
from rcc.benchmarks.fanoutqa.multihop import (
    audit_dependency_shards,
    decomposition_pageids,
    summarize_dependency_audits,
)
from rcc.benchmarks.fanoutqa.natural_panel import (
    natural_audit_body,
    natural_items,
    natural_page_texts,
    natural_source_rows,
    validate_natural_bundle,
    validate_natural_source_audit,
)
from rcc.benchmarks.fanoutqa.payload import read_payload, write_payload
from rcc.benchmarks.fanoutqa.prepare import (
    _PreparedPanelUnpickler,
    load_prepared_panel,
    prepared_source_commit,
)
from rcc.benchmarks.fanoutqa.qwen_serving import panel_registration_sha256
from rcc.benchmarks.fanoutqa.scoring import (
    DEFAULT_SCORER_VERSION,
    gold_leaf_groups,
    load_gold_patch,
    score_text_original,
    scorer_for_rows,
    scorer_for_version,
    validate_gold_patch,
)
from rcc.benchmarks.fanoutqa.scoring_equivalence import score_text_equivalent
from rcc.benchmarks.fanoutqa.source_audit import panel_prepared_paths
from rcc.benchmarks.fanoutqa.source_build import (
    append_construction_row,
    prepared_fixed_fields,
    seal_prepared_panel,
)
from rcc.benchmarks.fanoutqa.source_padding import ProbeItem
from rcc.benchmarks.fanoutqa.source_planning import source_page_provenance
from rcc.benchmarks.fanoutqa.source_topology import shard_fingerprints
from rcc.hardware.qwen_manifest import (
    placement_fingerprint,
    placements_record,
    provisional_placements,
    registered_policies,
)
from rcc.models.qwen import QWEN, QWEN_FAMILY
from rcc.run import identity
from rcc.run.barrier import SPLIT_FLEET_ISOLATION_PROFILE
from rcc.run.config import load_run_config
from rcc.run.plan import ResolutionContext, load_and_resolve, resolve_plan
from rcc.run.qwen.adapter import (
    QWEN_EXECUTION_ROSTER_FINGERPRINT,
    build_adapter,
)

# --- The loader: freeze integrity, the panel, the scorer.


def test_frozen_question_source_is_commit_pinned():
    assert FANOUTQA_DATASET_REVISION == "48da441031b16f83b2c71d478c45150c7dece613"
    assert FANOUTQA_DATASET_REVISION in FANOUTQA_DEV_URL
    assert "/main/" not in FANOUTQA_DEV_URL


def test_freeze_artifact_integrity():
    freeze = load_freeze()
    assert freeze["S"] == 60000
    assert freeze["census_version"] == 2, "v2 = recursive evidence walk, no ftfy"
    assert len(freeze["eligible_ids"]) == freeze["n_eligible"]
    assert len(freeze["dev_ids"]) == 30
    assert len(freeze["heldout_ids"]) == freeze["n_eligible"] - 30
    assert set(freeze["dev_ids"]).isdisjoint(freeze["heldout_ids"])
    assert set(freeze["dev_ids"]) | set(freeze["heldout_ids"]) == set(freeze["eligible_ids"])
    by_sha = sorted(freeze["eligible_ids"], key=lambda i: hashlib.sha256(i.encode()).hexdigest())
    assert freeze["dev_ids"] == by_sha[:30], "dev membership is ascending sha256, first 30"
    assert freeze["heldout_ids"] == by_sha[30:]
    assert len(freeze["page_tokens"]) == freeze["n_pages"]
    assert freeze["n_pages"] >= 1594, "recursive walk must cover the nested branches"
    assert "no ftfy" in freeze["content_pipeline"].lower()
    assert "recursive" in freeze["content_pipeline"].lower()

    # The held-out pool arithmetic the registered panel spends: 208 eligible,
    # 30 dev, 178 held out; the natural fifty are drawn from the held-out pool
    # in ascending sha256(qid), so the development items are reserved.
    heldout = freeze["heldout_ids"]
    assert (freeze["n_eligible"], len(freeze["dev_ids"]), len(heldout)) == (208, 30, 178)
    natural = list(FANOUTQA_NATURAL_DEV50.question_ids)
    assert set(natural).issubset(set(heldout))
    assert set(natural).isdisjoint(set(freeze["dev_ids"]))
    assert len(set(natural)) == 50
    assert natural == sorted(natural, key=lambda i: hashlib.sha256(i.encode()).hexdigest())
    # The sealed natural contract may never move.
    assert FANOUTQA_NATURAL_DEV50.scientific_identity_hash == (
        "50e0726fd210e85b512612fdbbab3f2ccdaa4356f031a2862e4be5ea35f460ed"
    )
    assert FANOUTQA_NATURAL_DEV50.question_index_sha256 == freeze["dev_json_sha256"]
    assert FANOUTQA_NATURAL_DEV50.max_model_len == 131_072

    # The buildable panel resolves by name, and an unregistered name is refused.
    assert build_cli.resolve_profile("fanoutqa-natural-dev50") is FANOUTQA_NATURAL_DEV50
    with pytest.raises(ValueError, match="unregistered benchmark profile"):
        build_cli.resolve_profile("fanoutqa-pending")


def test_load_questions_walks_nested_decomposition(tmp_path):
    """Nested fan-out branches carry evidence a flat walk would drop."""
    import json as _json

    from rcc.benchmarks.fanoutqa.data import load_questions

    rec = {
        "id": "q1",
        "question": "?",
        "answer": {"a": 1},
        "decomposition": [
            {"evidence": {"pageid": 1, "revid": 10, "title": "top"}},
            {
                "evidence": None,
                "decomposition": [
                    {"evidence": {"pageid": 2, "revid": 20, "title": "leafA"}},
                    {
                        "evidence": None,
                        "decomposition": [
                            {"evidence": {"pageid": 3, "revid": 30, "title": "leafB"}}
                        ],
                    },
                    {"evidence": {"pageid": 1, "revid": 10, "title": "top"}},
                ],
            },
        ],
    }
    path = tmp_path / "dev.json"
    path.write_text(_json.dumps([rec]))
    (question,) = load_questions(path, verify_sha=False)
    assert [p[0] for p in question.pages] == [1, 2, 3], "nested evidence must be walked, deduped"


def test_real_fanout_audit_requires_official_disjoint_complete_page_branches():
    raw = {
        "answer": "x",
        "decomposition": [
            {
                "id": "top",
                "depends_on": [],
                "evidence": {"pageid": 1, "revid": 10, "title": "a"},
            },
            {
                "id": "nested",
                "depends_on": ["top"],
                "evidence": None,
                "decomposition": [
                    {
                        "id": "left",
                        "depends_on": [],
                        "evidence": {"pageid": 2, "revid": 20, "title": "b"},
                    },
                    {
                        "id": "right",
                        "depends_on": ["left"],
                        "evidence": {"pageid": 3, "revid": 30, "title": "c"},
                    },
                ],
            },
        ],
    }
    question = Question(
        qid="real",
        question="?",
        pages=((1, 10, "a"), (2, 20, "b"), (3, 30, "c")),
        raw=raw,
    )
    plan = ShardPlan(
        shards=((1, 3), (2,)),
        kept={1: 1, 2: 1, 3: 1},
        ledger=(),
    )
    audit = real_fanout_structure_audit(question, plan, content_pageids=(1, 2, 3))
    assert audit["official_unique_pages"] == 3
    assert audit["max_branch_width"] == 2
    assert audit["worker_page_counts"] == [2, 1]
    assert audit["worker_pages_disjoint"] is True
    assert audit["dependency_edges"] == 2
    assert audit["dependency_edges_spanning_workers"] == 2
    assert audit["reasoning_shape"] == "explicit_cross_worker_dependency"

    duplicated = ShardPlan(
        shards=((1, 2), (2, 3)),
        kept=plan.kept,
        ledger=(),
    )
    with pytest.raises(RuntimeError, match="share a page"):
        real_fanout_structure_audit(question, duplicated, content_pageids=(1, 2, 3))

    # A flat one-page question is not a fan-out handoff at all.
    flat = Question(
        qid="flat",
        question="?",
        pages=((1, 10, "a"),),
        raw={
            "answer": "x",
            "decomposition": [
                {
                    "id": "only",
                    "depends_on": [],
                    "evidence": {"pageid": 1, "revid": 10, "title": "a"},
                }
            ],
        },
    )
    with pytest.raises(RuntimeError, match="not a valid real FanOutQA handoff"):
        real_fanout_structure_audit(
            flat, ShardPlan(shards=((1,), ()), kept={1: 1}, ledger=()), content_pageids=(1,)
        )


def _question(answer, qid="q"):
    return Question(qid=qid, question="?", pages=((1, 1, "t"),), raw={"id": qid, "answer": answer})


def test_answerability_audit_separates_question_given_removed_and_absent_leaves():
    question = Question(
        qid="q",
        question="Is Alpha involved?",
        pages=((1, 1, "t"),),
        raw={"answer": {"Alpha": "53.3", "Other": "404"}},
    )
    audit = evidence_answerability_audit(
        question,
        full_sources=("Other has an average value of 53.3.",),
        retained_sources=("Other remains available.",),
    )
    assert audit["n_gold_leaves"] == 4
    assert audit["n_question_given"] == 1
    assert audit["n_off_question_leaves"] == 3
    assert audit["n_locatable_in_full_source"] == 2
    assert audit["n_survives_construction"] == 1
    assert audit["n_removed_by_construction"] == 1
    assert audit["n_absent_from_full_source"] == 1
    assert {row["leaf"]: row["status"] for row in audit["leaves"]} == {
        "Alpha": "question_given",
        "Other": "survives_construction",
        "53.3": "removed_by_construction",
        "404": "absent_from_full_source",
    }


def test_gold_leaves_official_chain():
    """Dict keys count alongside values (the official denominator is len*2); bools are yes/no."""
    assert gold_leaves(_question({"a": "Right", "b": "Left"})) == ("a", "b", "Right", "Left")
    assert gold_leaves(_question(["x", {"y": 2}])) == ("x", "y", "2")
    assert gold_leaves(_question(61.6)) == ("61.6",)
    assert gold_leaves(_question(True)) == ("yes",)
    assert gold_leaves(_question(False)) == ("no",)


def test_score_text_loose_strict_and_numbers(tmp_path):
    q = _question({"The Phantom Menace": "$1.027 billion", "Ioniq 5": 63.2})
    text = "The Phantom Menace grossed $1.027 billion; the Ioniq 5 reached 63.2 mpge."
    scores = score_text(q, text)
    assert scores["loose"] == 1.0 and scores["strict"] == 1.0
    assert scores["n_leaves"] == 4.0, "two keys plus two values"
    partial = score_text(q, "the Ioniq 5 reached 63.2")
    assert partial["loose"] == pytest.approx(2 / 4)
    assert partial["strict"] == 0.0
    assert score_text(q, "")["loose"] == 0.0
    grouped = _question({"pop": 1200000})
    assert score_text(grouped, "the pop was 1,200,000 people")["loose"] == 1.0
    boundary = _question(["42"])
    assert score_text(boundary, "the value 420 appears")["loose"] == 0.0
    assert score_text(boundary, "the value 42 appears")["loose"] == 1.0
    # The pinned-page patch: a listed question accepts the page's literal
    # string, splits concatenated references into one required group each, and
    # refuses a drifted original.
    patch = load_gold_patch()
    for qid, entries in patch.items():
        for index, (original, groups) in entries.items():
            assert original and all(group and all(group) for group in groups), (qid, index)
    # These qids key gold_patch.json and equivalence_aliases.json.
    scotus = _question(["John Roberts", "Thomas Clarence"], qid="8e62c5c69ce8c58d")
    assert score_text(scotus, "John Roberts and Clarence Thomas")["strict"] == 1.0
    assert score_text(scotus, "John Roberts and Thomas Clarence")["strict"] == 0.0
    floyd = _question({"Pink Floyd": "Syd Barrett, David Gilmour, and Roger Waters"})
    floyd_patched = _question(floyd.raw["answer"], qid="d098623a8b75ec0a")
    assert gold_leaf_groups(floyd) == (
        ("Pink Floyd",),
        ("Syd Barrett, David Gilmour, and Roger Waters",),
    )
    assert score_text(floyd_patched, "Pink Floyd: Roger Waters")["strict"] == 0.0
    with pytest.raises(RuntimeError, match="gold patch names"):
        gold_leaf_groups(_question(["not", "the dataset"], qid="8e62c5c69ce8c58d"))
    with pytest.raises(RuntimeError, match="not on this panel"):
        validate_gold_patch([scotus])
    with pytest.raises(RuntimeError, match="does not name a leaf"):
        validate_gold_patch([_question(["only one leaf"], qid=qid) for qid in patch])


# --- The equivalence scorer's boundaries.


def _question_checks(answer: object, qid: str = "example") -> Question:
    return Question(qid, "?", (), {"id": qid, "answer": answer})


def _check_notation() -> None:
    accepted = [
        ("72", "72.0"),
        ("72.00", "72"),
        ("72", "72.0."),
        ("1200000", "1,200,000.00"),
        ("9,596,961 km2", "9,596,961 km²"),
        ("12 cm²", "12 cm2"),
        ("1 mm2", "1 mm²"),
        ("5 m2", "5 m²"),
        ("12km2", "12km²"),
    ]
    rejected = [
        ("72", "72.1"),
        ("72", "172.0"),
        ("72", "v72.0"),
        ("72", "72.0.1"),
        ("72", "72.0.foo"),
        ("72", "72.0kg"),
        ("1974", "1973-74"),
        ("1974", "1973\u201374"),
        ("12 km2", "12 m²"),
        ("12 km2", "12 km³"),
        ("somekm2", "somekm²"),
    ]
    for target, cases in [(1.0, accepted), (0.0, rejected)]:
        for reference, answer in cases:
            actual = score_text_equivalent(_question_checks(reference), answer)["strict"]
            assert actual == target, (reference, answer)
            assert score_text(_question_checks(reference), answer)["strict"] == target


def _check_directors() -> None:
    leaves = [f"required{i}" for i in range(9)] + ["Anthony Russo, Joe Russo"]
    question = _question_checks(leaves, "445e0f67be85856c")
    prefix = " ".join(leaves[:9]) + " "
    for names in [
        "Anthony Russo, Joe Russo",
        "Anthony Russo and Joe Russo",
        "Anthony and Joe Russo",
    ]:
        assert score_text_equivalent(question, prefix + names) == {
            "loose": 1.0,
            "strict": 1.0,
            "n_leaves": 10.0,
        }
    for names in ["Joe Russo", "Anthony Russo", "Anthony and Jim Russo"]:
        assert score_text_equivalent(question, prefix + names)["strict"] == 0.0
    with pytest.raises(ValueError, match="reference"):
        score_text_equivalent(_question_checks(["other"], "445e0f67be85856c"), "other")
    unrelated = _question_checks("Anthony Russo, Joe Russo")
    assert score_text_equivalent(unrelated, "Anthony and Joe Russo")["strict"] == 0.0


def _check_paper_scores() -> None:
    question = _question_checks("72")
    assert score_text(question, "72.0")["strict"] == 1.0
    assert score_text_original(question, "72.0")["strict"] == 0.0
    scotus = _question_checks(["John Roberts", "Thomas Clarence"], "8e62c5c69ce8c58d")
    assert score_text_equivalent(scotus, "John Roberts and Clarence Thomas")["strict"] == 1.0
    assert score_text_equivalent(scotus, "John Roberts and Thomas Clarence")["strict"] == 0.0
    assert score_text_original(scotus, "John Roberts and Clarence Thomas")["strict"] == 0.0


def _check_alias_validation(tmp_path: Path) -> None:
    payload = json.loads(scoring_equivalence._ALIASES_PATH.read_text())
    invalid = []
    for field, value in [
        ("alternatives", "Anthony and Joe Russo"),
        ("leaf_index", True),
        ("alternatives", [" "]),
        ("original", 1),
    ]:
        bad = copy.deepcopy(payload)
        bad["entries"][0][field] = value
        invalid.append(bad)
    invalid.append({**payload, "policy": "unknown"})
    invalid.append({**payload, "entries": {}})
    path = tmp_path / "aliases.json"
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(scoring_equivalence, "_ALIASES_PATH", path)
            for bad in invalid:
                path.write_text(json.dumps(bad))
                scoring_equivalence._aliases.cache_clear()
                with pytest.raises(ValueError):
                    score_text_equivalent(_question_checks("72"), "72")
    finally:
        scoring_equivalence._aliases.cache_clear()


def test_equivalence_policy(tmp_path: Path) -> None:
    """Exercise equivalent notation, required names, and the paper scores."""
    _check_notation()
    _check_directors()
    _check_paper_scores()
    _check_alias_validation(tmp_path)
    _check_scoring_versions()


def _check_scoring_versions() -> None:
    question = _question_checks("72")
    assert DEFAULT_SCORER_VERSION == "fanoutqa-equivalence-v1"
    scorer = scorer_for_version(DEFAULT_SCORER_VERSION)
    assert scorer is score_text
    assert scorer(question, "72.0")["strict"] == 1.0
    assert scorer_for_rows([{"scoring_version": DEFAULT_SCORER_VERSION}]) is scorer
    for rows in [
        [{}, {"scoring_version": DEFAULT_SCORER_VERSION}],
        [{}, {}],
        [{"scoring_version": "fanoutqa-string-proxy-loose-strict-v1"}],
        [{"scoring_version": "fanoutqa-gold-patch-v1"}],
        [{"scoring_version": "unknown"}],
        [{"scoring_version": None}],
        [{"scoring_version": []}],
    ]:
        with pytest.raises(ValueError, match="scoring version"):
            scorer_for_rows(rows)


# --- The natural panel and its bundle.


def test_natural_profile_contract() -> None:
    """The geometry field only enters the identity of a natural profile."""
    natural = FANOUTQA_NATURAL_DEV50
    exact = dataclasses.replace(natural, prompt_geometry="exact")
    assert not natural.exact_prompts and exact.exact_prompts
    assert "prompt_geometry" in natural.to_dict()
    assert "prompt_geometry" not in exact.to_dict()
    assert natural.admits_worker_prompt_tokens((40_015, 53_248, 1))
    assert not natural.admits_worker_prompt_tokens((53_249, 1, 1))
    assert not natural.admits_worker_prompt_tokens((1, 1))
    assert exact.admits_worker_prompt_tokens((53_248,) * 3)
    assert not exact.admits_worker_prompt_tokens((53_247, 53_248, 53_248))
    assert natural.rolled_rows(44_000) == 44_040
    assert len(set(natural.question_ids)) == 50
    with pytest.raises(ValueError, match="prompt geometry"):
        dataclasses.replace(natural, prompt_geometry="padded")


def test_natural_bundle_round_trip(tmp_path: Path) -> None:
    """A sealed bundle validates, tokenizes per family, audits, and refuses drift."""
    root = tmp_path / "bundle"
    profile = write_bundle(root)
    manifest = validate_natural_bundle(root, profile=profile)
    assert manifest["qids"] == ["q1"]
    items = natural_items(root, CharTokenizer(), profile=profile)
    assert [len(shard) for shard in items[0].shards] == [71, 67, 47]
    assert items[0].question_obj.raw["answer"] == ["Alpha", "Beta"]
    assert set(natural_page_texts(root, profile=profile)) == {11, 12, 13}
    rows = natural_source_rows(root, profile=profile)
    assert [row["pageid"] for row in rows] == [11, 12, 13]
    body = natural_audit_body(
        source_commit="0" * 40,
        config_fingerprint="c" * 16,
        items=items,
        construction_sha256="d" * 64,
        profile=profile,
    )
    audit = {
        **body,
        "audit_fingerprint": identity.fingerprint(body, identity.json_compact_legacy)[:16],
    }
    validate_natural_source_audit(
        audit,
        source_commit="0" * 40,
        config_fingerprint="c" * 16,
        ledger_sha256="d" * 64,
        profile=profile,
    )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        validate_natural_source_audit(
            audit,
            source_commit="1" * 40,
            config_fingerprint="c" * 16,
            ledger_sha256="d" * 64,
            profile=profile,
        )
    with pytest.raises(RuntimeError, match="outside the registered ceiling"):
        natural_audit_body(
            source_commit="0" * 40,
            config_fingerprint="c" * 16,
            items=items,
            construction_sha256="d" * 64,
            profile=dataclasses.replace(profile, worker_prompt_tokens=60),
        )
    item_path = root / "items" / "q1.json"
    item = json.loads(item_path.read_text(encoding="utf-8"))
    item["workers"][0]["text"] = "changed " + item["workers"][0]["text"]
    item_path.write_text(json.dumps(item), encoding="utf-8")
    with pytest.raises(RuntimeError, match="drifted"):
        validate_natural_bundle(root, profile=profile)


# --- Dependency and shard topology.


def _node(
    node_id: str,
    pageid: int | None,
    *,
    depends_on: tuple[str, ...] = (),
    decomposition: tuple[dict[str, object], ...] = (),
) -> dict[str, object]:
    evidence = None if pageid is None else {"pageid": pageid, "revid": pageid * 10}
    return {
        "id": node_id,
        "question": node_id,
        "answer": node_id,
        "depends_on": list(depends_on),
        "evidence": evidence,
        "decomposition": list(decomposition),
    }


def test_dependency_audit_maps_explicit_edges_across_workers():
    decomposition = (
        _node("root", 1),
        _node("left", 2, depends_on=("root",)),
        _node("right", 3, depends_on=("root",)),
    )

    audit = audit_dependency_shards(decomposition, ((1, 3), (2,)))

    assert audit.schema == "fanoutqa-multihop-audit-v2"
    assert audit.decomposition_nodes == 3
    assert audit.evidence_mentions == 3
    assert audit.cited_pageids == frozenset({1, 2, 3})
    assert audit.max_branch_width == 3
    assert audit.max_decomposition_depth == 1
    assert audit.dependency_edges == 2
    assert audit.dependency_chain_nodes == 2
    assert audit.dependency_edges_spanning_workers == 1
    assert audit.worker_evidence_mentions == (2, 1)
    assert audit.reasoning_shape == "explicit_cross_worker_dependency"


def test_dependency_audit_tracks_nested_composite_support():
    nested = (
        _node("leaf-a", 2),
        _node("leaf-b", 3, depends_on=("leaf-a",)),
    )
    decomposition = (
        _node("root", 1),
        _node("composite", None, depends_on=("root",), decomposition=nested),
    )

    audit = audit_dependency_shards(decomposition, ((1, 2), (3,)))

    assert decomposition_pageids(decomposition) == (1, 2, 3)
    assert audit.max_decomposition_depth == 2
    assert audit.dependency_edges == 2
    assert audit.dependency_chain_nodes == 2
    assert audit.dependency_edges_spanning_workers == 2
    assert audit.worker_dependency_incidence == (2, 2)


def test_dependency_audit_labels_parallel_fanout_without_overclaiming():
    decomposition = tuple(_node(f"leaf-{index}", index) for index in range(1, 6))

    audit = audit_dependency_shards(decomposition, ((1, 3, 5), (2, 4)))

    assert audit.dependency_edges == 0
    assert audit.dependency_chain_nodes == 1
    assert audit.dependency_edges_spanning_workers == 0
    assert audit.reasoning_shape == "parallel_fanout"


@pytest.mark.parametrize(
    ("decomposition", "match"),
    [
        ((_node("a", 1, depends_on=("missing",)), _node("b", 2)), "unknown node"),
        (
            (_node("a", 1, depends_on=("b",)), _node("b", 2, depends_on=("a",))),
            "cycle",
        ),
        ((_node("same", 1), _node("same", 2)), "duplicate decomposition node"),
    ],
)
def test_dependency_audit_rejects_invalid_graphs(decomposition, match):
    with pytest.raises(RuntimeError, match=match):
        audit_dependency_shards(decomposition, ((1,), (2,)))


def test_dependency_audit_summary_separates_explicit_and_parallel_items():
    explicit = audit_dependency_shards(
        (_node("root", 1), _node("leaf", 2, depends_on=("root",))),
        ((1,), (2,)),
    )
    parallel = audit_dependency_shards(
        tuple(_node(f"p-{index}", index) for index in range(3, 8)),
        ((3, 5, 7), (4, 6)),
    )

    summary = summarize_dependency_audits((explicit, parallel))

    assert summary["items"] == 2
    assert summary["explicit_dependency_items"] == 1
    assert summary["parallel_fanout_items"] == 1
    assert summary["cross_worker_dependency_items"] == 1
    assert summary["min_unique_pages"] == 2
    assert summary["max_unique_pages"] == 5


def _synthetic_prepared_items() -> tuple[ProbeItem, ...]:
    """Build the sealed production roster as inert, tiny prepared items."""
    items: list[ProbeItem] = []
    for index, qid in enumerate(FANOUTQA_NATURAL_DEV50.question_ids, start=1):
        question = Question(
            qid=qid,
            question=f"question {index}?",
            pages=((index, 1000 + index, f"Page {index}"),),
            raw={"answer": {"leaf": str(index)}},
        )
        items.append(
            ProbeItem(
                qid=qid,
                question=question.question,
                shards=tuple(
                    (index, index + 1) for _ in range(FANOUTQA_NATURAL_DEV50.workers_per_item)
                ),
                question_obj=question,
            )
        )
    return tuple(items)


def test_payload_manifest_hashes_every_referenced_file(tmp_path: Path):

    cut = tmp_path / "cut.pt"
    cut.write_bytes(b"latent-cut")
    payload = write_payload(
        tmp_path / "payloads",
        qid="q1",
        files={"cut": cut},
        meta={"keep_sha": "abc"},
    )
    manifest = read_payload(payload, qid="q1")
    assert manifest["meta"] == {"keep_sha": "abc"}
    assert Path(manifest["files"]["cut"]["path"]) == cut

    cut.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        read_payload(payload, qid="q1")


def test_the_prepared_panel_pins_its_commit_restricts_unpickling_and_round_trips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RCC_FANOUT_PREPARED_SOURCE_COMMIT", FANOUTQA_NATURAL_DEV50.source_commit)
    assert (
        prepared_source_commit("a" * 40, profile=FANOUTQA_NATURAL_DEV50)
        == FANOUTQA_NATURAL_DEV50.source_commit
    )
    monkeypatch.setenv("RCC_FANOUT_PREPARED_SOURCE_COMMIT", "b" * 40)
    with pytest.raises(RuntimeError, match=FANOUTQA_NATURAL_DEV50.profile_id):
        prepared_source_commit("a" * 40, profile=FANOUTQA_NATURAL_DEV50)
    monkeypatch.delenv("RCC_FANOUT_PREPARED_SOURCE_COMMIT")

    loader = _PreparedPanelUnpickler(io.BytesIO())
    assert loader.find_class("rcc.benchmarks.fanoutqa.source_padding", "ProbeItem") is ProbeItem
    assert loader.find_class("rcc.benchmarks.fanoutqa.panel", "Question") is Question
    with pytest.raises(pickle.UnpicklingError, match="forbidden global"):
        loader.find_class("os", "system")
    with pytest.raises(pickle.UnpicklingError, match="forbidden global"):
        loader.find_class("builtins", "eval")

    # The write half round-trips through that loader: seal a synthetic panel,
    # then rehash the manifest, unpickle under the restricted globals and
    # revalidate the ledger. Only the source-audit rebuild is stubbed.
    items = _synthetic_prepared_items()
    old_item = _synthetic_prepared_items()[0]
    old_item.__dict__.pop("native_prompt_artifacts")
    assert (
        _PreparedPanelUnpickler(io.BytesIO(pickle.dumps(old_item))).load().native_prompt_artifacts
        is None
    )
    audit = {
        "audit_fingerprint": FANOUTQA_NATURAL_DEV50.source_audit_fingerprint,
        "shards": shard_fingerprints(items),
    }
    run_root = tmp_path / "run"
    fixed = prepared_fixed_fields(
        source_commit=FANOUTQA_NATURAL_DEV50.source_commit,
        source_audit=audit,
        profile=FANOUTQA_NATURAL_DEV50,
    )
    ledger = panel_prepared_paths(run_root, FANOUTQA_NATURAL_DEV50)[2]
    for item in items:
        for pageid, revid, _title in item.question_obj.pages:
            append_construction_row(
                ledger,
                {
                    **source_page_provenance(
                        qid=item.qid,
                        pageid=pageid,
                        revid=revid,
                        raw_text=f"raw {pageid}",
                        canonical_text=f"canonical {pageid}",
                        actual_tokens=11,
                        census_tokens=11,
                    ),
                    **fixed,
                    "banked_at_unix": 0.0,
                },
            )
    manifest = seal_prepared_panel(
        run_root,
        source_commit=FANOUTQA_NATURAL_DEV50.source_commit,
        items=items,
        source_audit=audit,
        profile=FANOUTQA_NATURAL_DEV50,
    )
    monkeypatch.setattr(prepare_module, "validate_source_audit", lambda *_a, **_k: audit)
    loaded_items, loaded_manifest = load_prepared_panel(
        run_root,
        source_commit=FANOUTQA_NATURAL_DEV50.source_commit,
        profile=FANOUTQA_NATURAL_DEV50,
    )
    assert loaded_manifest == manifest
    assert tuple(item.qid for item in loaded_items) == FANOUTQA_NATURAL_DEV50.question_ids
    assert manifest["prepared_schema"] == "fanoutqa-m3-serving-prepared-v2"
    assert manifest["panel_registration_sha"] == FANOUTQA_NATURAL_DEV50.panel_registration_sha256
    assert manifest["config_fingerprint"] == FANOUTQA_NATURAL_DEV50.prepared_config_fingerprint
    # The bytes just sealed sit at the profile-keyed paths every reader names.
    keyed = run_root / "prepared" / "profiles" / FANOUTQA_NATURAL_DEV50.profile_id
    assert panel_prepared_paths(run_root, FANOUTQA_NATURAL_DEV50) == (
        keyed / "items.pkl",
        keyed / "items.manifest.json",
        keyed / "construction.jsonl",
    )


def test_registered_direct_placements_and_shared_qwen_isolation(tmp_path: Path):
    adapter = build_adapter(QWEN_FAMILY, profile=FANOUTQA_NATURAL_DEV50)
    registered = provisional_placements(family=QWEN_FAMILY)
    # The registry is the only placement authority: an empty run root resolves
    # the same table the adapter registers, with no minted file in between.
    assert (
        adapter.placements.resolve(tmp_path, panel="production", source_commit="a" * 40)
        == adapter.placements.registered
        == registered
    )
    # The lane baseline: the family roster cut to this panel's arms. The lane
    # implements twelve more seats for the chain's two budget laws, six
    # re-ranked and six bounded, and no FanOutQA arm is one of them.
    roster = registered_policies(QWEN_FAMILY, FANOUTQA_NATURAL_DEV50)
    assert len(roster) == 11 and len(QWEN.physical_arms) == 23
    assert not any("rerank" in policy or "bounded" in policy for policy in roster)
    assert tuple(registered) == roster
    assert [(registered[policy].producers, registered[policy].receivers) for policy in roster] == [
        (0, 8),
        (0, 8),
        (6, 2),
        (6, 2),
        (4, 4),
        (5, 3),
        (5, 3),
        (5, 3),
        (5, 3),
        (5, 3),
        (5, 3),
    ]
    # What the run records is exactly what it resolved, under its own plan.
    record = placements_record(QWEN_FAMILY, FANOUTQA_NATURAL_DEV50, "c" * 64)
    assert record["version"] == "route-registry-placements-v1"
    assert record["plan_fingerprint"] == "c" * 64
    assert set(record["placements"]) == set(roster)
    assert record["placements"][roster[0]] == {
        "workers": 8,
        "producers": 0,
        "receivers": 8,
        "fused": False,
    }
    assert placement_fingerprint(QWEN_FAMILY, FANOUTQA_NATURAL_DEV50) == "51a1b083db344e1c"
    assert adapter.registration()["isolation_profile"] == SPLIT_FLEET_ISOLATION_PROFILE
    # Signed with the FanOutQA roster alone, so the chain's latent arms in the
    # lane's physical roster never move it, however many the lane implements.
    assert QWEN_EXECUTION_ROSTER_FINGERPRINT == (
        "eece64221d20836ee582075871e1f28757eaf6838dba5fd65eb07e33fe3b2a2a"
    )


# --- The plan identities of the six shipped configs.

_ROOT = Path(__file__).resolve().parents[1]


_COMMIT = "0" * 40


_FIELDS = (
    "run_id",
    "output_uri",
    "run_fingerprint",
    "execution_fingerprint",
    "scientific_fingerprint",
)


SNAPSHOT: dict[str, tuple[str, ...]] = {
    "gemma-fanoutqa-natural-dev50.toml": (
        "gemma-fanoutqa-m3-v1-gemma-fanoutqa-natural-dev50-fleet-v16",
        "runs/gemma-fanoutqa-m3/gemma-fanoutqa-m3-v1-gemma-fanoutqa-natural-dev50-fleet-v16",
        "554cd042f02c4d8af280a9829b63830058d2a53ed897cba815e7bbcf1abdf13a",
        "f3a5ad2666eafd0ea084e63bf6fb9d2f39ea7f10d7f3bead8c00ff1e01f393a2",
        "50e0726fd210e85b512612fdbbab3f2ccdaa4356f031a2862e4be5ea35f460ed",
    ),
    "ministral-fanoutqa-natural-dev50.toml": (
        "ministral-fanoutqa-m3-v1-ministral-fanoutqa-natural-dev50-target-v1",
        "runs/ministral-fanoutqa-m3/ministral-fanoutqa-m3-v1-ministral-fanoutqa-natural-dev50-target-v1",
        "ad8e640afdb665aee625f127efd3c2e58898b87cf47ab08a1149de460b5040d0",
        "cf4d5c61c4e24e921122edb306f5b0dac1f94b320944e732871a10b8865189f6",
        "50e0726fd210e85b512612fdbbab3f2ccdaa4356f031a2862e4be5ea35f460ed",
    ),
    "nemotron-fanoutqa-natural-dev50.toml": (
        "nemotron-fanoutqa-m3-n15-v1-nemotron-fanoutqa-natural-dev50",
        "runs/fanoutqa/nemotron-fanoutqa-m3-n15-v1-nemotron-fanoutqa-natural-dev50",
        "1952046b1bb9d91167ccc6e823d36b7199ea2160ab0bb8a095c53c4e11035e8c",
        "4085458459718964697416240ed1f127de1a20cb7ce9bc0ae18ef65e157d13c9",
        "9995ce77140f1bd28b559ae908fa74019481c9eb04aa0ceb003e785ab2ff0a1a",
    ),
    "qwen-fanoutqa-natural-dev50-judger-ablation.toml": (
        "qwen-fanoutqa-m3-n15-v1-qwen-fanoutqa-natural-dev50-judger",
        "runs/fanoutqa/qwen-fanoutqa-m3-n15-v1-qwen-fanoutqa-natural-dev50-judger",
        "082d3d91989cc2df27399ab7512eb742f83881cb5cdc26bd19612caeb2a7a434",
        "8026350a6c9f1db6b472d9ccdf46eb6de1f59427a6bdddc11ce1ddd3eae1d918",
        "9cf73dba266cb3131bab6158ea6d15c5832414d0426f0b3c72ef4ee79d493f5b",
    ),
    # The same ablation on the seven latent arms alone: 1050 rows.
    "qwen-fanoutqa-natural-dev50-latent-judger-ablation.toml": (
        "qwen-fanoutqa-m3-n15-v1-qwen-fanoutqa-natural-dev50-latent-judger",
        "runs/fanoutqa/qwen-fanoutqa-m3-n15-v1-qwen-fanoutqa-natural-dev50-latent-judger",
        "71e7f99a48f78b33871ab781d87022ff7e6b46300159e3f5419c328fbcec2722",
        "8172751e2e3168a8abc68ed48fbed08a762ea04f2b83a38d3ec422f522940220",
        "f74dd3d40cbff03b4210640a85084b37691cd1f5dfc46df6590f19e2e58865bd",
    ),
    "qwen-fanoutqa-natural-dev50.toml": (
        "qwen-fanoutqa-m3-n15-v1-qwen-fanoutqa-natural-dev50",
        "runs/fanoutqa/qwen-fanoutqa-m3-n15-v1-qwen-fanoutqa-natural-dev50",
        "efc30bb05e4ae43d0a252635d576a2d95fdaedfc5521f3bd68d55bca17b9c8f5",
        "2a4f8baf58711392c8a8d53b0c220d2e2a280c8573573089b1de4e1cbfe4b665",
        "50e0726fd210e85b512612fdbbab3f2ccdaa4356f031a2862e4be5ea35f460ed",
    ),
}


def test_every_fanoutqa_plan_identity_is_byte_identical_to_the_snapshot():
    configs = sorted(path.name for path in (_ROOT / "configs").glob("*fanoutqa*.toml"))
    assert configs == sorted(SNAPSHOT), "a FanOutQA config appeared or vanished"
    for name, expected in SNAPSHOT.items():
        plan = load_and_resolve(_ROOT / "configs" / name, git_commit=_COMMIT)
        observed = (
            plan.run_id,
            plan.output_uri,
            plan.run_identity_hash,
            plan.execution_identity_hash,
            plan.scientific_identity_hash,
        )
        for field, value, pinned in zip(_FIELDS, observed, expected, strict=True):
            assert value == pinned, f"{name}: {field} moved"
    # The judger ablation: the question rides the plan, a blank one and a
    # resident lane are both refused, and the registered run carries none.
    question = "Why was Le Chaton Fat denied a small-business loan?"
    ablation_path = _ROOT / "configs" / "qwen-fanoutqa-natural-dev50-judger-ablation.toml"
    base = load_and_resolve(
        _ROOT / "configs" / "qwen-fanoutqa-natural-dev50.toml", git_commit=_COMMIT
    )
    ablation = load_and_resolve(ablation_path, git_commit=_COMMIT)
    assert (base.benchmark.judger_question, ablation.benchmark.judger_question) == (None, question)
    assert ablation.scientific_identity_hash != base.scientific_identity_hash
    context = ResolutionContext(git_commit=_COMMIT)
    with pytest.raises(ValueError, match="carry text"):
        resolve_plan(
            dataclasses.replace(load_run_config(ablation_path), judger_question=" "), context
        )
    gemma = load_run_config(_ROOT / "configs" / "gemma-fanoutqa-natural-dev50.toml")
    with pytest.raises(ValueError, match="FanOutQA route lanes only"):
        resolve_plan(dataclasses.replace(gemma, judger_question=question), context)
    chain = load_run_config(_ROOT / "configs" / "qwen-longbench-coa-easy50-text-n50.toml")
    with pytest.raises(ValueError, match="FanOutQA route lanes only"):
        resolve_plan(dataclasses.replace(chain, judger_question=question), context)
    # The latent cut is the natural profile with only its roster moved: the
    # same fifty items, config fingerprint, and every answer and report seed;
    # its own serving registration digest, which lists the policies.
    natural, latent = FANOUTQA_NATURAL_DEV50, FANOUTQA_NATURAL_DEV50_LATENT
    assert [arm.channel for arm in latent.arms] == ["latent"] * 7
    assert tuple(arm.arm_id for arm in latent.arms) == tuple(
        arm.arm_id for arm in natural.arms if arm.channel == "latent"
    )
    assert dict(latent.sealed_qwen_policies) == {
        arm.arm_id: dict(natural.sealed_qwen_policies)[arm.arm_id] for arm in latent.arms
    }
    assert latent.panel_registration_sha256 == panel_registration_sha256(latent, family=QWEN_FAMILY)
    assert latent.prepared_artifact_sha256 != natural.prepared_artifact_sha256
    same = (
        "prepared_config_fingerprint",
        "source_audit_fingerprint",
        "question_ids",
        "answer_seed_namespace",
        "report_seed_base",
        "sample_tags",
        "worker_prompt_tokens",
        "latent_steps",
        "span_width",
        "ratios",
        "declared_passes",
    )
    assert all(getattr(latent, name) == getattr(natural, name) for name in same)
    assert all(
        latent.answer_seeds(qid, arm.arm_id) == natural.answer_seeds(qid, arm.arm_id)
        and latent.report_seeds(qid, tag) == natural.report_seeds(qid, tag)
        for qid in natural.question_ids
        for arm in latent.arms
        for tag in natural.sample_tags
    )
    cut = load_and_resolve(
        _ROOT / "configs" / "qwen-fanoutqa-natural-dev50-latent-judger-ablation.toml",
        git_commit=_COMMIT,
    )
    assert cut.expected_rows.to_dict() == {"items": 50, "arms": 7, "seeds": 3, "total": 1050}
    assert cut.benchmark.judger_question == question
