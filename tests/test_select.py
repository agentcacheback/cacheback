"""The selection interface: `Select` over a scorer, the span schedules, and the scorers."""

import itertools
from types import SimpleNamespace

import pytest
import torch
from tests.conftest import random_ids, tiny_config
from transformers import Qwen3ForCausalLM

from rcc import Axis, KVCache, Pipeline, Select, factored, handoff, span_schedule
from rcc.transforms.select import grad_scores, h2o_scores, snapkv_scores, span_keep
from rcc.transforms.select.scorers import (
    _pool_cluster,
    selfq_scores,
    snapkv_query_scores,
)
from rcc.transforms.select.select import select_keep
from rcc.transforms.select.spans import _span_bounds


def test_snapkv_scores_finite_shape(tiny_model):
    ids = random_ids(length=12, seed=1)
    scores = snapkv_scores(tiny_model, ids, obs=4)
    assert scores.shape == (12,)
    assert torch.isfinite(scores).all()


def test_h2o_scores_finite_shape(tiny_model):
    ids = random_ids(length=12, seed=2)
    scores = h2o_scores(tiny_model, ids)
    assert scores.shape == (12,)
    assert torch.isfinite(scores).all()


class _NoAttentionModel:
    """Stand-in for a model on an SDPA / FlashAttention path (returns no attentions)."""

    def __call__(self, **_kwargs):
        return SimpleNamespace(attentions=None)


def test_scorer_raises_without_attentions():
    with pytest.raises(ValueError, match="eager"):
        snapkv_scores(_NoAttentionModel(), random_ids(length=6))


def test_grad_scores_finite_shape_sign_cpu(tiny_model):
    scores = grad_scores(
        tiny_model, random_ids(length=12, seed=20), random_ids(length=5, seed=21), draft_tokens=4
    )
    assert scores.shape == (12,)
    assert scores.dtype == torch.float32
    assert scores.device.type == "cpu"
    assert torch.isfinite(scores).all()
    assert (scores >= 0).all()


def test_grad_scores_deterministic(tiny_model):
    ctx = random_ids(length=10, seed=22)
    prompt = random_ids(length=4, seed=23)
    first = grad_scores(tiny_model, ctx, prompt, draft_tokens=4)
    second = grad_scores(tiny_model, ctx, prompt, draft_tokens=4)
    assert torch.allclose(first, second)


def test_grad_scores_leaves_model_grads_none(tiny_model):
    # tiny_model is shared by every test in the run, so the scorer must
    # differentiate w.r.t. detached cache leaves via torch.autograd.grad and
    # never call .backward() on a graph that reaches model parameters.
    grad_scores(
        tiny_model, random_ids(length=10, seed=24), random_ids(length=4, seed=25), draft_tokens=4
    )
    assert all(p.grad is None for p in tiny_model.parameters())


def test_grad_scores_reenables_grad_under_no_grad(tiny_model):
    # A caller scoring inside inference code holds no_grad; the scorer needs
    # gradients internally, so it must re-enable grad locally rather than inherit
    # the disabled outer state.
    with torch.no_grad():
        scores = grad_scores(
            tiny_model,
            random_ids(length=10, seed=26),
            random_ids(length=4, seed=27),
            draft_tokens=4,
        )
    assert scores.shape == (10,)
    assert torch.isfinite(scores).all()


def test_grad_scores_pool_kernel_one_ok_even_raises(tiny_model):
    ctx = random_ids(length=12, seed=30)
    prompt = random_ids(length=4, seed=31)
    unpooled = grad_scores(tiny_model, ctx, prompt, draft_tokens=4, pool_kernel=1)
    assert unpooled.shape == (12,)
    with pytest.raises(ValueError, match="odd"):
        grad_scores(tiny_model, ctx, prompt, draft_tokens=4, pool_kernel=4)


def test_grad_scores_rejects_nonpositive_draft(tiny_model):
    with pytest.raises(ValueError):
        grad_scores(
            tiny_model, random_ids(length=8, seed=32), random_ids(length=4, seed=33), draft_tokens=0
        )


def test_grad_scores_eos_empty_draft_raises(tiny_model, monkeypatch):
    # Make the model's first greedy continuation token its eos id, so the eos-stopped
    # draft is length zero; the scorer must refuse rather than substitute. monkeypatch
    # (not setattr) restores the shared model afterwards.
    ctx = random_ids(length=10, seed=34)
    prompt = random_ids(length=4, seed=35)
    first = handoff(KVCache.from_prefill(tiny_model, ctx), tiny_model, prompt, max_new_tokens=1)
    forced = int(first[0, 0])
    monkeypatch.setattr(tiny_model.generation_config, "eos_token_id", forced)
    with pytest.raises(ValueError, match="empty answer draft"):
        grad_scores(tiny_model, ctx, prompt, draft_tokens=4)


def test_grad_scores_runs_under_sdpa():
    # The scorer has no eager requirement, so an SDPA model scores. SDPA
    # cannot stand in for the attention-vote scorers' raise-on-None path,
    # because transformers falls back to eager math under output_attentions.
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(tiny_config("sdpa")).eval()
    ctx = random_ids(length=10, seed=36)
    scores = grad_scores(model, ctx, random_ids(length=4, seed=37), draft_tokens=2)
    assert scores.shape == (10,)
    assert torch.isfinite(scores).all()


def _draft_logprob(model, cache, prompt, draft):
    """Oracle for the attributed quantity: teacher-forced draft log-likelihood."""
    n, q, k = cache.length, int(prompt.shape[1]), int(draft.shape[1])
    ids = torch.cat([prompt, draft], dim=1)
    with torch.no_grad():
        out = model(
            input_ids=ids,
            attention_mask=torch.ones(1, n + q + k, dtype=torch.long),
            past_key_values=cache.to_hf_cache(),
            position_ids=torch.arange(n, n + q + k).unsqueeze(0),
            use_cache=True,
            return_dict=True,
        )
    log_probs = torch.log_softmax(out.logits[0, q - 1 : q + k - 1, :].float(), dim=-1)
    return float(log_probs[torch.arange(k), draft[0]].sum())


def test_grad_scores_matches_finite_difference(tiny_model):
    # Semantic pin: the score is |grad times activation| per (layer, head,
    # side, column), summed, so the oracle must perturb one slot at a time.
    # Scaling a whole column measures the signed cross-layer sum instead.
    ctx = random_ids(length=8, seed=38)
    prompt = random_ids(length=4, seed=39)
    cache = KVCache.from_prefill(tiny_model, ctx)
    draft = handoff(cache, tiny_model, prompt, max_new_tokens=4, stop_at_eos=True)
    assert int(draft.shape[1]) == 4  # the scorer attributes this exact draft
    scores = grad_scores(tiny_model, ctx, prompt, draft_tokens=4, pool_kernel=1)
    base = _draft_logprob(tiny_model, cache, prompt, draft)
    eps = 1e-2
    fd = torch.zeros(cache.length)
    for col in range(cache.length):
        for side in ("keys", "values"):
            for layer in range(cache.layers):
                for head in range(cache.kv_heads):
                    keys, values = cache.keys.clone(), cache.values.clone()
                    (keys if side == "keys" else values)[layer, head, col, :] *= 1 + eps
                    scaled = KVCache(
                        keys=keys, values=values, positions=cache.positions, rope=cache.rope
                    )
                    fd[col] += abs(_draft_logprob(tiny_model, scaled, prompt, draft) - base) / eps
    corr = torch.corrcoef(torch.stack([scores, fd]))[0, 1]
    assert float(corr) > 0.99
    assert int(scores.argmax()) == int(fd.argmax())


def test_select_keep_sink_always_kept():
    scores = torch.arange(10, dtype=torch.float)  # position 0 scores lowest
    keep = select_keep(scores, budget=3)
    assert 0 in keep  # sink survives even though it scores lowest
    assert keep == sorted(keep)
    assert len(keep) == 3


def test_select_keep_clamps_budget_to_length():
    scores = torch.randn(5)
    assert select_keep(scores, budget=99) == [0, 1, 2, 3, 4]


def test_select_keep_rejects_sink_larger_than_budget():
    # The sink is never dropped, so a sink the budget cannot hold must raise:
    # returning it whole would exceed the budget and misprice the compression.
    with pytest.raises(ValueError, match="sink"):
        select_keep(torch.randn(8), budget=2, sink=(0, 1, 2))
    # Out-of-range sink positions do not count against the budget.
    assert len(select_keep(torch.randn(8), budget=2, sink=(0, 90))) == 2


def test_select_ratio_halves_length_and_meters(tiny_model):
    cache = KVCache.from_prefill(tiny_model, random_ids(length=12, seed=3))
    scores = torch.arange(12, dtype=torch.float)
    out = Select(scores, ratio=2).apply(cache)

    assert out.length == 6
    assert out.meters().resident_bytes < cache.meters().resident_bytes
    assert out.meters().wire_bytes < cache.meters().wire_bytes

    ratio = factored(cache, out)
    assert ratio.per_axis[Axis.LENGTH] == pytest.approx(2.0)

    positions = out.positions.tolist()
    assert 0 in positions  # sink kept
    assert positions == sorted(positions)  # original indices, sorted
    assert positions == [0, 7, 8, 9, 10, 11]  # sink + the six highest scores


def test_select_budget_selects_exact_count(tiny_model):
    cache = KVCache.from_prefill(tiny_model, random_ids(length=10, seed=8))
    out = Select(torch.randn(10), budget=4).apply(cache)
    assert out.length == 4
    assert 0 in out.positions.tolist()


def test_select_raises_on_missing_rope():
    keys = torch.randn(2, 2, 6, 4)
    cache = KVCache(keys=keys, values=keys.clone(), positions=torch.arange(6))
    with pytest.raises(ValueError, match="rope"):
        Select(torch.randn(6), budget=3).apply(cache)


def test_select_raises_on_scores_length_mismatch(tiny_model):
    cache = KVCache.from_prefill(tiny_model, random_ids(length=8, seed=9))
    with pytest.raises(ValueError, match="scores length"):
        Select(torch.randn(7), budget=3).apply(cache)


def test_select_requires_exactly_one_of_ratio_or_budget():
    with pytest.raises(ValueError, match="exactly one"):
        Select(torch.randn(4))
    with pytest.raises(ValueError, match="exactly one"):
        Select(torch.randn(4), ratio=2, budget=2)


def test_select_does_not_fire_meter_violation(tiny_model):
    cache = KVCache.from_prefill(tiny_model, random_ids(length=10, seed=10))
    # Only wire/resident bytes move, both declared, so the Pipeline assertion passes.
    out = Pipeline((Select(torch.randn(10), ratio=2),)).apply(cache)
    assert out.length == 5


@torch.no_grad()
def _decode(model, past, prefix_len, query_ids, n_new):
    q_len = query_ids.shape[1]
    total = prefix_len + q_len
    out = model(
        input_ids=query_ids,
        past_key_values=past,
        position_ids=torch.arange(prefix_len, total).unsqueeze(0),
        attention_mask=torch.ones(1, total, dtype=torch.long),
        use_cache=True,
        return_dict=True,
    )
    past = out.past_key_values
    nxt = out.logits[:, -1:].argmax(-1)
    tokens = []
    for _ in range(n_new):
        tokens.append(int(nxt[0, 0]))
        out = model(
            input_ids=nxt,
            past_key_values=past,
            position_ids=torch.tensor([[total]]),
            attention_mask=torch.ones(1, total + 1, dtype=torch.long),
            use_cache=True,
            return_dict=True,
        )
        past = out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1)
        total += 1
    return tokens


@pytest.mark.parametrize("schedule", [select_keep, span_schedule(span_size=4)])
def test_select_keep_everything_is_exact(tiny_model, schedule):
    # budget == length keeps every token; relocation is the identity, so a
    # decode continuation from the selected cache matches the full-cache
    # continuation token for token, which is the relocation-identity check.
    ctx = random_ids(length=14, seed=11)
    query = random_ids(length=4, seed=12)
    cache = KVCache.from_prefill(tiny_model, ctx)
    out = Select(torch.zeros(cache.length), budget=cache.length, schedule=schedule).apply(cache)

    assert out.length == cache.length
    assert torch.equal(out.keys, cache.keys)  # no rotation applied at full budget

    full_tokens = _decode(tiny_model, cache.to_hf_cache(), cache.length, query, n_new=8)
    kept_tokens = _decode(tiny_model, out.to_hf_cache(), out.length, query, n_new=8)
    assert kept_tokens == full_tokens


@pytest.mark.parametrize("schedule", [select_keep, span_schedule(span_size=4)])
def test_select_relocated_cache_decodes_deterministically(tiny_model, schedule):
    # The non-identity relocation path (need_move True) yields a usable cache:
    # greedy decode is deterministic across two calls.
    cache = KVCache.from_prefill(tiny_model, random_ids(length=16, seed=13))
    out = Select(torch.arange(16, dtype=torch.float), ratio=2, schedule=schedule).apply(cache)
    query = random_ids(length=3, seed=14)
    first = _decode(tiny_model, out.to_hf_cache(), out.length, query, n_new=6)
    second = _decode(tiny_model, out.to_hf_cache(), out.length, query, n_new=6)
    assert first == second


@pytest.mark.parametrize("schedule", [select_keep, span_schedule(span_size=4)])
def test_select_composes_on_relocated_cache(tiny_model, schedule):
    # After one real subset selection the provenance map is not `arange`, but
    # the physical rotation of slot i is always i, so a second keep-everything
    # Select is a bit-exact no-op on the tensors.
    cache = KVCache.from_prefill(tiny_model, random_ids(length=16, seed=15))
    once = Select(torch.arange(16, dtype=torch.float), ratio=2, schedule=schedule).apply(cache)
    assert once.positions.tolist() != list(range(once.length))  # provenance, not arange

    twice = Select(torch.zeros(once.length), budget=once.length).apply(once)

    assert torch.equal(twice.keys, once.keys)
    assert torch.equal(twice.values, once.values)
    assert torch.equal(twice.positions, once.positions)


def test_nan_scores_refuse_instead_of_outranking_infinity():
    scores = torch.tensor([float("inf"), 1.0, float("nan"), 0.0])
    with pytest.raises(ValueError, match="NaN"):
        select_keep(scores, budget=1, sink=())


def test_a_column_shaped_score_vector_still_ranks():
    flat = select_keep(torch.tensor([0.1, 0.9, 0.5]), budget=2, sink=())
    column = select_keep(torch.tensor([[0.1], [0.9], [0.5]]), budget=2, sink=())
    assert flat == column


def test_a_multi_column_score_tensor_refuses_loudly():
    with pytest.raises(ValueError, match="one vector"):
        select_keep(torch.ones(3, 2), budget=2, sink=())


TINY_VOCAB = 512


def test_span_pooling_equality_hand_computed():
    # n=8, span_size=4 -> spans (0,4),(4,8); the high span is (4,8). budget 4 with
    # the sink at 0 keeps the sink plus a 3-wide window around the high span's peak.
    scores = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    keep = span_keep(scores, budget=4, sink=(0,), span_size=4)
    assert keep == [0, 4, 5, 6]


def test_whole_span_kept_without_boundary_fragment():
    # budget admits the entire high span plus the sink, so no boundary fragment.
    scores = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    keep = span_keep(scores, budget=5, sink=(0,), span_size=4)
    assert set(keep) == {0, 4, 5, 6, 7}


def test_partial_window_is_single_contiguous_run_inside_one_span():
    n, span_size = 10, 5  # spans (0,5),(5,10)
    scores = torch.zeros(n)
    scores[5:10] = torch.tensor([1.0, 1.0, 3.0, 1.0, 1.0])  # peak at index 7
    keep = span_keep(scores, budget=3, sink=(0,), span_size=span_size)
    non_sink = [p for p in keep if p != 0]
    assert non_sink == list(range(non_sink[0], non_sink[0] + len(non_sink)))  # contiguous
    assert all(5 <= p < 10 for p in non_sink)  # entirely inside the high span
    assert len(non_sink) == 2  # room == budget - sink


def test_exact_budget_across_ragged_grid():
    n = 17
    scores = torch.randn(n, generator=torch.Generator().manual_seed(1))
    for span_size in (3, 4, 5, 8, 32):  # several do not divide n evenly
        for budget in (1, 2, 5, 9, 16, 17, 40):
            keep = span_keep(scores, budget=budget, sink=(0,), span_size=span_size)
            assert len(keep) == min(budget, n)
            assert keep == sorted(keep)


def test_sink_larger_than_budget_is_rejected():
    scores = torch.arange(8, dtype=torch.float)
    with pytest.raises(ValueError, match="sink"):
        span_keep(scores, budget=2, sink=(0, 1, 2), span_size=4)


def test_full_budget_is_identity_keep():
    n = 12
    scores = torch.randn(n, generator=torch.Generator().manual_seed(2))
    assert span_keep(scores, budget=n, span_size=5) == list(range(n))
    assert span_keep(scores, budget=999, span_size=5) == list(range(n))


def test_span_schedule_matches_span_keep_partial_binding():
    scores = torch.randn(20, generator=torch.Generator().manual_seed(3))
    sched = span_schedule(6)
    for budget in (3, 7, 20):
        assert sched(scores, budget, (0,)) == span_keep(scores, budget, (0,), span_size=6)


def test_span_bounds_tile_exactly():
    for n in (1, 5, 8, 17, 33):
        for span_size in (1, 3, 4, 32):
            bounds = _span_bounds(n, span_size)
            assert bounds[0][0] == 0
            assert bounds[-1][1] == n
            for (a_lo, a_hi), (b_lo, _b_hi) in itertools.pairwise(bounds):
                assert a_hi == b_lo  # contiguous, no gaps or overlaps
                assert a_hi > a_lo


def test_span_select_positions_match_span_keep(tiny_model):
    cache = KVCache.from_prefill(tiny_model, random_ids(length=12, seed=4))
    scores = torch.arange(12, dtype=torch.float)
    keep = span_keep(scores, budget=6, sink=(0,), span_size=4)
    out = Select(scores, budget=6, schedule=span_schedule(span_size=4)).apply(cache)
    assert out.positions.tolist() == keep  # provenance is the span keep set
    assert out.length == 6


@torch.no_grad()
def _joint_vote(model, context_ids, prompt_ids, row_lo, row_hi):
    """Reference reduction: sum attention from rows [row_lo, row_hi) over context cols."""
    joint = torch.cat([context_ids, prompt_ids], dim=1)
    total = joint.shape[1]
    n_ctx = context_ids.shape[1]
    pos = torch.arange(total).unsqueeze(0)
    out = model(
        input_ids=joint,
        position_ids=pos,
        use_cache=False,
        return_dict=True,
        output_attentions=True,
    )
    score = torch.zeros(n_ctx)
    for a in out.attentions:
        score += a[0].mean(dim=0)[row_lo:row_hi, :n_ctx].sum(dim=0).float().cpu()
    return score


def test_snapkv_query_found_path_uses_bare_question_window(tiny_model):
    ctx = random_ids(length=10, seed=20)
    question = torch.tensor([[101, 102, 103]])
    prompt = torch.cat([torch.tensor([[201, 202]]), question, torch.tensor([[203]])], dim=1)
    off, q_len = 2, 3
    result = snapkv_query_scores(tiny_model, ctx, prompt, question)
    expected = _pool_cluster(_joint_vote(tiny_model, ctx, prompt, 10 + off, 10 + off + q_len), 7)
    assert result.shape == (10,)
    assert torch.isfinite(result).all()
    assert torch.allclose(result, expected)  # located window, not the full-prompt fallback


def test_snapkv_query_seam_trim_path(tiny_model):
    ctx = random_ids(length=10, seed=21)
    # the full question is absent, but the interior tokens [101,102] appear at offset 1.
    question = torch.tensor([[100, 101, 102, 103]])
    prompt = torch.tensor([[50, 101, 102, 51]])
    result = snapkv_query_scores(tiny_model, ctx, prompt, question)
    expected = _pool_cluster(_joint_vote(tiny_model, ctx, prompt, 10 + 1, 10 + 1 + 2), 7)
    assert torch.isfinite(result).all()
    assert torch.allclose(result, expected)


def test_snapkv_query_all_prompt_fallback(tiny_model):
    ctx = random_ids(length=10, seed=22)
    question = torch.tensor([[300, 301, 302]])  # neither full nor interior appear
    prompt = torch.tensor([[10, 11, 12, 13]])
    total = 10 + prompt.shape[1]
    result = snapkv_query_scores(tiny_model, ctx, prompt, question)
    expected = _pool_cluster(_joint_vote(tiny_model, ctx, prompt, 10, total), 7)
    assert torch.isfinite(result).all()
    assert torch.allclose(result, expected)  # all prompt rows voted


class _StubTokenizer:
    """Minimal invertible tokenizer: ids <-> space-joined integer strings.

    Enough surface for `selfq_scores`, with a deterministic invertible round
    trip so the two-call determinism check means something.
    """

    eos_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(t)) for t in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        return messages[-1]["content"]

    def __call__(
        self,
        text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=False,
        max_length=None,
    ):
        toks = [int(t) % TINY_VOCAB for t in text.split() if t.lstrip("-").isdigit()]
        if not toks:
            toks = [1]
        if max_length is not None:
            toks = toks[:max_length]
        return {"input_ids": torch.tensor([toks])}


def test_selfq_scores_finite_shape(tiny_model):
    ids = random_ids(length=12, seed=30)
    scores = selfq_scores(tiny_model, _StubTokenizer(), ids, n_questions=2, max_new_tokens=8)
    assert scores.shape == (12,)
    assert torch.isfinite(scores).all()


def test_selfq_scores_deterministic_across_calls(tiny_model):
    ids = random_ids(length=12, seed=31)
    tok = _StubTokenizer()
    first = selfq_scores(tiny_model, tok, ids, n_questions=2, max_new_tokens=8)
    second = selfq_scores(tiny_model, tok, ids, n_questions=2, max_new_tokens=8)
    assert torch.equal(first, second)
