"""The baselines CacheBack is compared against: H2O, StreamingLLM, ChunkKV, and KVzip."""

from __future__ import annotations

import math

import pytest
import torch
from tests.conftest import random_ids
from torch.nn import functional as nnf

from rcc import h2o_scores
from rcc.latent.cache import cache_kv, cache_length
from rcc.transforms.select.baselines.chunkkv import (
    chunk_means,
    chunk_position_scores,
    memory_chunkkv_scores,
)
from rcc.transforms.select.baselines.chunkkv_native import (
    native_window_scores,
    window_stream_scores,
)
from rcc.transforms.select.baselines.h2o import (
    chunk_accumulated_scores,
    h2o_row_scores,
    memory_h2o_scores,
)
from rcc.transforms.select.baselines.kvzip import (
    memory_kvzip_scores,
    native_reconstruction_scores,
    project_position_scores,
    reconstruction_inputs,
    reconstruction_stream_scores,
)
from rcc.transforms.select.baselines.streaming import streaming_scores
from rcc.transforms.select.query_support.fixed_spans import locbench_rn_keep
from rcc.transforms.select.spans import span_keep

# --- H2O: streamed accumulated attention, parity with the eager scorer, chunk invariance.

# The two paths differ only in float32 accumulation order. The measured
# deviation on the tiny fixture is 4.8e-7 absolute at scores near 6.7, so this
# bound has a decade of headroom and none over a real disagreement.
PARITY_ATOL = 5e-6


PARITY_RTOL = 1e-6


def _prefilled(model, context: torch.Tensor, latent: int):
    """Prefill `context` plus `latent` extra rows, the prompt-plus-latent cache shape."""
    tail = random_ids(length=latent, seed=7) if latent else None
    ids = context if tail is None else torch.cat((context, tail), dim=1)
    with torch.no_grad():
        return model(input_ids=ids, use_cache=True).past_key_values


def _hand_weights() -> torch.Tensor:
    # 2 query heads, 3 query rows, 4 key columns; already softmaxed per row.
    return torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.2, 0.2, 0.6, 0.0]],
            [[1.0, 0.0, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0], [0.4, 0.4, 0.1, 0.1]],
        ]
    )


def test_row_reduction_is_the_eager_head_mean_of_the_row_column_sum() -> None:
    weights = _hand_weights()

    scores = h2o_row_scores(weights)

    manual = [
        sum(float(weights[head, row, column]) for head in range(2) for row in range(3)) / 2
        for column in range(4)
    ]
    assert scores.shape == (4,)
    assert scores.dtype == torch.float32
    torch.testing.assert_close(scores, torch.tensor(manual), atol=1e-7, rtol=0)
    # written out so the frozen rule is readable, not merely rederived
    torch.testing.assert_close(
        scores,
        torch.tensor([1.6, 1.0, 0.35, 0.05]),
        atol=1e-6,
        rtol=0,
    )


def test_row_reduction_averages_over_every_head_of_a_partial_group() -> None:
    weights = _hand_weights()

    group = h2o_row_scores(weights[:1], query_heads=2)
    other = h2o_row_scores(weights[1:], query_heads=2)

    torch.testing.assert_close(group + other, h2o_row_scores(weights), atol=1e-7, rtol=0)


def test_row_reduction_rejects_shapes_and_divisors_it_cannot_honour() -> None:
    weights = _hand_weights()
    with pytest.raises(ValueError):
        h2o_row_scores(weights[0])
    with pytest.raises(ValueError):
        h2o_row_scores(weights, query_heads=1)
    with pytest.raises(ValueError):
        h2o_row_scores(weights[:, :0])


def _reference_full_scores(
    query: torch.Tensor, key: torch.Tensor, *, scaling: float
) -> torch.Tensor:
    """The unchunked rule: one full causal map, softmaxed per row, then reduced."""
    rows, key_length = int(query.shape[2]), int(key.shape[2])
    memory = key_length - rows
    groups = int(query.shape[1]) // int(key.shape[1])
    keys = torch.repeat_interleave(key, groups, dim=1)
    logits = torch.matmul(query, keys.transpose(-2, -1)) * scaling
    index = torch.arange(rows)
    logits[..., memory:] += torch.where(
        index.unsqueeze(0) <= index.unsqueeze(1),
        torch.zeros((), dtype=logits.dtype),
        torch.full((), torch.finfo(logits.dtype).min, dtype=logits.dtype),
    )
    return h2o_row_scores(torch.softmax(logits, dim=-1)[0])


def test_chunk_scores_reproduce_one_full_causal_map() -> None:
    torch.manual_seed(21)
    query = torch.randn(1, 6, 5, 4, dtype=torch.float64)
    key = torch.randn(1, 3, 9, 4, dtype=torch.float64)

    actual = chunk_accumulated_scores(query, key, scaling=1 / math.sqrt(4))
    expected = _reference_full_scores(query, key, scaling=1 / math.sqrt(4))

    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("chunk", [1, 2, 3, 5])
def test_streaming_query_rows_in_chunks_changes_no_value(chunk: int) -> None:
    torch.manual_seed(22)
    rows, head_dim = 5, 4
    query = torch.randn(1, 4, rows, head_dim, dtype=torch.float64)
    key = torch.randn(1, 2, rows, head_dim, dtype=torch.float64)
    scaling = 1 / math.sqrt(head_dim)

    streamed = torch.zeros(rows, dtype=torch.float64)
    for start in range(0, rows, chunk):
        end = min(start + chunk, rows)
        # the chunk's rows attend to the whole key prefix, never a window
        part = chunk_accumulated_scores(query[:, :, start:end], key[:, :, :end], scaling=scaling)
        streamed[:end] += part

    expected = _reference_full_scores(query, key, scaling=scaling)
    torch.testing.assert_close(streamed, expected, atol=1e-12, rtol=0)


def test_chunk_scores_reject_geometry_they_cannot_group() -> None:
    query = torch.randn(1, 4, 3, 4)
    key = torch.randn(1, 3, 6, 4)
    with pytest.raises(ValueError):
        chunk_accumulated_scores(query, key, scaling=0.5)
    with pytest.raises(ValueError):
        chunk_accumulated_scores(query, torch.randn(1, 2, 2, 4), scaling=0.5)
    with pytest.raises(ValueError):
        chunk_accumulated_scores(query[0], torch.randn(1, 2, 6, 4), scaling=0.5)
    with pytest.raises(ValueError):
        chunk_accumulated_scores(query, torch.randn(2, 2, 6, 4), scaling=0.5)


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 17])
def test_streamed_scores_match_the_reference_eager_h2o_scorer(tiny_model, chunk_size: int) -> None:
    context = random_ids(length=17, seed=920)
    past = _prefilled(tiny_model, context, latent=4)

    streamed = memory_h2o_scores(tiny_model, past, context, chunk_size=chunk_size)
    eager = h2o_scores(tiny_model, context)

    assert streamed.shape == (21,)
    torch.testing.assert_close(streamed[:17], eager, atol=PARITY_ATOL, rtol=PARITY_RTOL)
    # the latent tail holds no prompt row, so it accumulates nothing
    assert torch.equal(streamed[17:], torch.zeros(4))


def test_capture_is_deterministic_finite_and_full_length(tiny_model) -> None:
    context = random_ids(length=12, seed=922)
    past = _prefilled(tiny_model, context, latent=3)

    first = memory_h2o_scores(tiny_model, past, context, chunk_size=5)
    second = memory_h2o_scores(tiny_model, past, context, chunk_size=5)

    assert first.shape == (cache_length(past),)
    assert first.dtype == torch.float32
    assert first.device.type == "cpu"
    assert bool(torch.isfinite(first).all())
    assert bool(torch.all(first[:12] > 0))
    assert torch.equal(first, second)


def test_capture_leaves_the_handoff_cache_and_attn_impl_untouched(tiny_model) -> None:
    context = random_ids(length=12, seed=923)
    past = _prefilled(tiny_model, context, latent=3)
    snapshot = [(keys.clone(), values.clone()) for keys, values in cache_kv(past)]
    impl_before = str(tiny_model.config._attn_implementation)

    memory_h2o_scores(tiny_model, past, context, chunk_size=5)

    assert cache_length(past) == 15
    for (keys, values), (want_keys, want_values) in zip(cache_kv(past), snapshot, strict=True):
        assert torch.equal(keys, want_keys)
        assert torch.equal(values, want_values)
    assert str(tiny_model.config._attn_implementation) == impl_before


def test_a_failing_chunk_restores_the_attn_impl_and_surfaces_the_error(
    tiny_model, monkeypatch
) -> None:
    context = random_ids(length=12, seed=924)
    past = _prefilled(tiny_model, context, latent=3)
    impl_before = str(tiny_model.config._attn_implementation)

    def failing_forward(*args, **kwargs):
        raise RuntimeError("injected chunk failure")

    monkeypatch.setattr(tiny_model.model, "forward", failing_forward)

    with pytest.raises(RuntimeError, match="injected chunk failure"):
        memory_h2o_scores(tiny_model, past, context, chunk_size=5)

    assert str(tiny_model.config._attn_implementation) == impl_before
    assert cache_length(past) == 15


def test_scorer_rejects_inputs_it_cannot_score(tiny_model) -> None:
    context = random_ids(length=12, seed=925)
    past = _prefilled(tiny_model, context, latent=3)
    with pytest.raises(ValueError):
        memory_h2o_scores(tiny_model, past, context[0], chunk_size=4)
    with pytest.raises(ValueError):
        memory_h2o_scores(tiny_model, past, context, chunk_size=0)
    with pytest.raises(ValueError):
        memory_h2o_scores(tiny_model, past, random_ids(length=20, seed=926), chunk_size=4)


# --- StreamingLLM: the recency score.


def protected_positions(length: int, protected_tail: int) -> tuple[int, ...]:
    """Sink 0 plus the last `protected_tail` positions, the pipeline's protected set."""
    tail_start = max(1, length - protected_tail)
    return (0, *range(tail_start, length))


def test_the_score_is_pure_recency():
    scores = streaming_scores(5)
    assert torch.equal(scores, torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]))
    assert scores.dtype == torch.float32
    assert streaming_scores(0).numel() == 0
    with pytest.raises(ValueError, match="non-negative"):
        streaming_scores(-1)


@pytest.mark.parametrize("ratio", (2, 64))
def test_the_keep_is_the_sink_plus_the_most_recent_spans(ratio: int):
    """The native sink-plus-window pattern under the exact-budget wrapper:
    everything kept outside the protected set is a trailing suffix of spans."""
    length, latent_steps = 4136, 40
    protected = protected_positions(length, latent_steps)
    keep = locbench_rn_keep(
        streaming_scores(length),
        float(ratio),
        protected=protected,
        span_size=32,
    )
    assert set(protected) <= set(keep)
    assert len(keep) == math.ceil(length / ratio)
    evidence = sorted(set(keep) - set(protected))
    # recency: the kept evidence is one contiguous trailing block
    assert evidence == list(range(evidence[0], evidence[-1] + 1))
    assert evidence[-1] == length - latent_steps - 1


# --- ChunkKV: chunk scoring, broadcast, projection, capture on the tiny model.


def rn_keep(
    scores: torch.Tensor,
    ratio: int,
    *,
    protected: tuple[int, ...] = (0,),
    span_size: int = 32,
) -> list[int]:
    """The near-equal span schedule at the exact `ceil(length / ratio)` budget."""
    budget = max(1, math.ceil(int(scores.shape[0]) / ratio))
    return span_keep(scores, budget, tuple(sorted(set(protected))), span_size=span_size)


def _upstream_chunk_scores(global_scores: torch.Tensor, chunk_length: int) -> torch.Tensor:
    """A literal transcription of kvpress `presses/chunkkv_press.py::ChunkKVPress.compress`.

    Public source pin: github.com/NVIDIA/kvpress at commit
    83066022418379439f6ae7f0c53197298bb5e36c, the chunk-score block only.
    """
    kv_len = global_scores.shape[-1]
    num_complete_chunks = kv_len // chunk_length
    remaining_tokens = kv_len % chunk_length
    if num_complete_chunks > 0:
        main_scores = global_scores[..., : num_complete_chunks * chunk_length]
        main_chunk_scores = main_scores.sum(dim=1).view(-1, num_complete_chunks, chunk_length)
        main_chunk_scores = main_chunk_scores.mean(dim=-1)
    else:
        main_chunk_scores = torch.empty((global_scores.shape[0], 0), device=global_scores.device)
    if remaining_tokens > 0:
        remaining_scores = global_scores[..., -remaining_tokens:]
        remaining_chunk_score = remaining_scores.sum(dim=1).mean(dim=-1, keepdim=True)
        chunk_scores = torch.cat([main_chunk_scores, remaining_chunk_score], dim=-1)
    else:
        chunk_scores = main_chunk_scores
    return chunk_scores


def _upstream_snapkv_score(
    attn_weights: torch.Tensor,
    *,
    bsz: int,
    num_key_value_heads: int,
    num_key_value_groups: int,
    k_len: int,
    window_size: int,
    kernel_size: int,
) -> torch.Tensor:
    """A literal transcription of kvpress `presses/snapkv_press.py::SnapKVPress.score`.

    Pinned to github.com/NVIDIA/kvpress commit
    42175e729984f08532bb070df37abd81ceefb494, from the softmaxed branch onward.
    """
    scores = attn_weights.mean(dim=-2)
    scores = nnf.avg_pool1d(scores, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
    scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, k_len - window_size)
    scores = scores.mean(2)
    scores = nnf.pad(scores, (0, window_size), value=scores.max().item() + 1)
    return scores


def _dyadic_streams() -> torch.Tensor:
    # 2 KV heads over 10 positions, every value an exact binary fraction so the
    # native sum and the frozen mean can be compared with zero tolerance.
    return torch.tensor(
        [
            [
                [0.25, 0.5, 0.75, 1.0, 0.125, 0.375, 0.625, 0.875, 0.25, 0.5],
                [1.0, 0.25, 0.5, 0.125, 0.75, 0.25, 0.5, 0.375, 0.125, 0.75],
            ]
        ]
    )


def test_chunk_scores_match_the_official_chunk_aggregation() -> None:
    streams = _dyadic_streams()
    kv_heads = int(streams.shape[1])

    ours = chunk_means(project_position_scores(streams), chunk_size=4)
    theirs = _upstream_chunk_scores(streams, 4)[0]

    assert ours.shape == theirs.shape == (3,)
    # the native reducer over KV heads is the sum; the frozen projection is the
    # mean, so the two differ by exactly the constant head count
    torch.testing.assert_close(ours * kv_heads, theirs, atol=0, rtol=0)
    torch.testing.assert_close(
        theirs,
        torch.tensor([1.09375, 0.96875, 0.8125]),
        atol=0,
        rtol=0,
    )


def test_every_position_in_a_chunk_carries_the_chunk_score() -> None:
    scores = torch.tensor([1.0, 3.0, 2.0, 6.0, 8.0, 4.0, 5.0])

    broadcast = chunk_position_scores(scores, chunk_size=3)
    means = chunk_means(scores, chunk_size=3)

    assert broadcast.shape == scores.shape
    torch.testing.assert_close(
        broadcast,
        torch.tensor([2.0, 2.0, 2.0, 6.0, 6.0, 6.0, 5.0]),
        atol=0,
        rtol=0,
    )
    for index, mean in enumerate(means.tolist()):
        members = broadcast[index * 3 : index * 3 + 3]
        assert members.tolist() == [mean] * int(members.shape[0])


def test_broadcast_keeps_whole_chunks_under_a_chunk_wide_span_ranking() -> None:
    # the on-grid pair: the shipped chunk_size default 16 against the
    # registered default span width 16
    torch.manual_seed(21)
    scores = chunk_position_scores(torch.rand(256), chunk_size=16)

    keep = locbench_rn_keep(scores, 4.0, protected=protected_positions(256, 0), span_size=16)

    # width 16 makes every span exactly one chunk, so the span ranking the keep
    # law applies is the native chunk ranking
    torch.testing.assert_close(scores.view(16, 16).mean(dim=-1), chunk_means(scores, chunk_size=16))
    kept = set(keep)
    partial = [
        chunk for chunk in range(16) if 0 < len(kept & set(range(chunk * 16, chunk * 16 + 16))) < 16
    ]
    # only the protected sink's chunk and the single exact-fill chunk can be
    # partial; every other kept chunk survives whole
    assert len(partial) <= 2
    assert any(set(range(chunk * 16, chunk * 16 + 16)) <= kept for chunk in range(16))


def test_a_narrower_span_width_slices_chunks_instead_of_keeping_them_whole() -> None:
    """At span width 8 the span, not the chunk, is the unit that survives whole.

    At chunk_size 16 against span width 8 the broadcast score is a 16-token box
    filter, so the keep is whole 8-token spans and a chunk can survive in part.
    """
    torch.manual_seed(23)
    scores = chunk_position_scores(torch.rand(256), chunk_size=16)

    keep = locbench_rn_keep(scores, 8.0, protected=protected_positions(256, 0), span_size=8)

    kept = set(keep)
    sliced = [
        chunk for chunk in range(8) if 0 < len(kept & set(range(chunk * 32, chunk * 32 + 32))) < 32
    ]
    # the budget is one chunk's worth of tokens spread over four-times-narrower
    # spans, so at least one chunk is kept in part: the whole-chunk unit does
    # not hold at this width
    assert sliced
    # the unit that stays whole here is the span: apart from the sink's own
    # span and the single exact-fill span, every kept span survives entire
    partial_spans = [
        start for start in range(0, 256, 8) if 0 < len(kept & set(range(start, start + 8))) < 8
    ]
    assert len(partial_spans) <= 2


def _hand_weights_chunkkv() -> torch.Tensor:
    torch.manual_seed(24)
    return torch.softmax(torch.randn(1, 4, 3, 11), dim=-1)


def test_window_stream_scores_match_the_official_snapkv_reduction() -> None:
    weights = _hand_weights_chunkkv()

    ours = window_stream_scores(weights[0], window_size=3, kv_groups=2, kernel_size=5)
    theirs = _upstream_snapkv_score(
        weights[..., -3:, :-3],
        bsz=1,
        num_key_value_heads=2,
        num_key_value_groups=2,
        k_len=11,
        window_size=3,
        kernel_size=5,
    )[0]

    assert ours.shape == theirs.shape == (2, 11)
    torch.testing.assert_close(ours, theirs, atol=0, rtol=0)


def test_window_stream_scores_floor_the_observation_window_above_every_score() -> None:
    weights = _hand_weights_chunkkv()

    scores = window_stream_scores(weights[0], window_size=3, kv_groups=2, kernel_size=5)

    scored, window = scores[:, :-3], scores[:, -3:]
    assert float(window.min()) > float(scored.max())
    assert float(window.min()) == float(window.max())


def test_window_stream_scores_reject_geometry_they_cannot_group() -> None:
    weights = _hand_weights_chunkkv()[0]
    with pytest.raises(ValueError):
        window_stream_scores(weights, window_size=3, kv_groups=3, kernel_size=5)
    with pytest.raises(ValueError):
        window_stream_scores(weights, window_size=11, kv_groups=2, kernel_size=5)
    with pytest.raises(ValueError):
        window_stream_scores(weights, window_size=3, kv_groups=2, kernel_size=4)
    with pytest.raises(ValueError):
        window_stream_scores(weights[0], window_size=3, kv_groups=2, kernel_size=5)


def _reference_window_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    scaling: float,
    window_size: int,
    kernel_size: int,
) -> torch.Tensor:
    """The unchunked observation-window score: one full softmax over every key."""
    kv_heads, k_len = int(key.shape[1]), int(key.shape[2])
    groups = int(query.shape[1]) // kv_heads
    keys = torch.repeat_interleave(key, groups, dim=1)
    logits = torch.matmul(query, keys.transpose(-2, -1)) * scaling
    mask = torch.triu(torch.ones_like(logits) * float("-inf"), diagonal=k_len - window_size + 1)
    weights = torch.softmax(logits + mask, dim=-1)
    return torch.stack(
        [
            window_stream_scores(
                weights[batch], window_size=window_size, kv_groups=groups, kernel_size=kernel_size
            )
            for batch in range(int(query.shape[0]))
        ]
    )


@pytest.mark.parametrize("column_chunk", [1, 3, 4096])
def test_chunked_capture_agrees_with_one_full_softmax(column_chunk: int) -> None:
    torch.manual_seed(25)
    window_size, prefix, head_dim = 4, 13, 8
    query = torch.randn(1, 4, window_size, head_dim)
    key = torch.randn(1, 2, prefix + window_size, head_dim)

    ours = native_window_scores(
        query,
        key,
        scaling=1 / math.sqrt(head_dim),
        window_size=window_size,
        kernel_size=5,
        column_chunk=column_chunk,
    )
    theirs = _reference_window_scores(
        query,
        key,
        scaling=1 / math.sqrt(head_dim),
        window_size=window_size,
        kernel_size=5,
    )

    assert ours.shape == theirs.shape == (1, 2, prefix + window_size)
    torch.testing.assert_close(ours, theirs, atol=1e-6, rtol=0)


def test_chunkkv_keep_never_exceeds_the_exact_budget() -> None:
    """Hold the budget law under the plateaus only a chunk-tied score produces.

    `chunk_position_scores` broadcasts one value per chunk, so the ranking
    arrives as plateaus of equal scores and the budget overflows at an edge.
    """
    torch.manual_seed(26)
    length = 1025
    scores = chunk_position_scores(project_position_scores(torch.rand(4, 3, length)), chunk_size=10)
    ratio, latent_steps = 8, 6
    budget = max(1, math.ceil(length / ratio))
    protected = set(protected_positions(length, latent_steps))

    for width in (None, 16):
        if width is None:
            keep = rn_keep(scores, ratio, protected=tuple(protected), span_size=32)
        else:
            keep = locbench_rn_keep(
                scores, float(ratio), protected=tuple(protected), span_size=width
            )
        assert len(keep) <= budget
        assert len(keep) == len(set(keep))
        assert keep == sorted(keep)
        assert all(0 <= position < length for position in keep)
        assert protected <= set(keep)


def test_tiny_capture_is_deterministic_and_restores_the_cache(tiny_model) -> None:
    context = random_ids(length=18, seed=921)
    with torch.no_grad():
        past = tiny_model(input_ids=context, use_cache=True).past_key_values
    entry_length = cache_length(past)
    snapshot = _cache_snapshot(past)
    impl_before = str(tiny_model.config._attn_implementation)

    first = memory_chunkkv_scores(
        tiny_model, past, context[:, :16], None, chunk_size=3, window_size=4
    )
    second = memory_chunkkv_scores(
        tiny_model, past, context[:, :16], None, chunk_size=3, window_size=4
    )

    assert first.shape == (entry_length,)
    assert first.dtype == torch.float32
    assert first.device.type == "cpu"
    assert torch.isfinite(first).all()
    assert torch.equal(first[16:], torch.zeros(2))
    assert bool(torch.all(first[:16] > 0))
    assert torch.equal(first, second)
    assert cache_length(past) == entry_length
    _assert_cache_bytes_equal(past, snapshot)
    assert str(tiny_model.config._attn_implementation) == impl_before
    for start in range(0, 15, 3):
        members = first[start : start + 3]
        assert float(members.min()) == float(members.max())


def _cache_snapshot(past) -> list[tuple[torch.Tensor, torch.Tensor]]:
    from rcc.latent.cache import cache_kv

    return [(keys.clone(), values.clone()) for keys, values in cache_kv(past)]


def _assert_cache_bytes_equal(past, snapshot) -> None:
    from rcc.latent.cache import cache_kv

    for (keys, values), (expected_keys, expected_values) in zip(
        cache_kv(past), snapshot, strict=True
    ):
        assert torch.equal(keys, expected_keys)
        assert torch.equal(values, expected_values)


def test_capture_failure_restores_the_cache_and_surfaces_the_error(tiny_model, monkeypatch) -> None:
    context = random_ids(length=18, seed=924)
    with torch.no_grad():
        past = tiny_model(input_ids=context, use_cache=True).past_key_values
    snapshot = _cache_snapshot(past)
    impl_before = str(tiny_model.config._attn_implementation)

    def failing_forward(*args, **kwargs):
        raise RuntimeError("injected capture failure")

    monkeypatch.setattr(tiny_model.model, "forward", failing_forward)

    with pytest.raises(RuntimeError, match="injected capture failure"):
        memory_chunkkv_scores(tiny_model, past, context[:, :16], None, chunk_size=3, window_size=4)

    assert cache_length(past) == 18
    _assert_cache_bytes_equal(past, snapshot)
    assert str(tiny_model.config._attn_implementation) == impl_before


def test_capture_refuses_a_window_that_leaves_no_scored_prompt(tiny_model) -> None:
    context = random_ids(length=8, seed=925)
    with torch.no_grad():
        past = tiny_model(input_ids=context, use_cache=True).past_key_values

    with pytest.raises(ValueError):
        memory_chunkkv_scores(tiny_model, past, context, None, chunk_size=3, window_size=8)
    with pytest.raises(ValueError):
        memory_chunkkv_scores(tiny_model, past, context, None, chunk_size=0, window_size=4)


# --- KVzip: reconstruction scores, projection, capture on the tiny model.


class _Tokenizer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str,
        add_special_tokens: bool,
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        assert not add_special_tokens
        self.calls.append(text)
        ids = [(ord(character) % 400) + 1 for character in text]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def test_position_projection_is_arithmetic_mean_not_any_stream_max() -> None:
    native = torch.tensor(
        [
            [[0.9, 0.1], [0.1, 0.5]],
            [[0.1, 0.5], [0.1, 0.5]],
        ]
    )
    projected = project_position_scores(native)

    assert torch.allclose(projected, torch.tensor([0.3, 0.4]))
    assert not torch.equal(projected, native.amax(dim=(0, 1)))
    # the lowered stream was never the maximum at position 0, so the forbidden
    # reducer cannot see the change at all; the frozen mean drops
    blind = torch.tensor([[[1.0, 2.0]], [[3.0, 0.0]]])
    lowered = blind.clone()
    lowered[0, 0, 0] = 0.0
    assert torch.equal(blind.amax(dim=(0, 1)), lowered.amax(dim=(0, 1)))
    assert float(project_position_scores(lowered)[0]) < float(project_position_scores(blind)[0])


def test_reconstruction_inputs_freeze_the_qwen3_chunk_protocol() -> None:
    tokenizer = _Tokenizer()
    context = torch.arange(10).unsqueeze(0)
    chunks = reconstruction_inputs(
        context,
        tokenizer,
        chunk_size=4,
        previous_suffix_size=2,
    )

    assert [(start, end) for start, end, _ in chunks] == [(1, 5), (5, 9), (9, 10)]
    assert tokenizer.calls.count("\n\nRepeat the previous context exactly.") == 1
    assert (
        tokenizer.calls.count("\n\nRepeat the part of the previous context exactly, starting with ")
        == 2
    )
    assert tokenizer.calls.count("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n") == 1
    postfix_length = len("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert torch.equal(
        chunks[1][2][:, -(postfix_length + 6) : -(postfix_length + 4)],
        context[:, 3:5],
    )
    assert torch.equal(
        chunks[2][2][:, -(postfix_length + 3) : -(postfix_length + 1)],
        context[:, 7:9],
    )
    assert torch.equal(chunks[0][2][:, -4:], context[:, 1:5])
    assert torch.equal(chunks[2][2][:, -1:], context[:, 9:10])


def test_tiny_qwen3_capture_is_deterministic_and_restores_the_cache(tiny_model) -> None:
    context = random_ids(length=12, seed=911)
    with torch.no_grad():
        past = tiny_model(input_ids=context, use_cache=True).past_key_values
    entry_length = cache_length(past)
    snapshot = _cache_snapshot(past)
    impl_before = str(tiny_model.config._attn_implementation)
    tokenizer = _Tokenizer()

    first = memory_kvzip_scores(
        tiny_model,
        past,
        context[:, :10],
        tokenizer,
        chunk_size=4,
    )
    second = memory_kvzip_scores(
        tiny_model,
        past,
        context[:, :10],
        tokenizer,
        chunk_size=4,
    )

    assert first.shape == (entry_length,)
    assert first.dtype == torch.float32
    assert first.device.type == "cpu"
    assert torch.isfinite(first).all()
    assert first[0] == 0
    assert bool(torch.all(first[1:10] > 0))
    assert torch.equal(first[10:], torch.zeros(2))
    assert torch.equal(first, second)
    assert cache_length(past) == entry_length
    _assert_cache_bytes_equal(past, snapshot)
    assert str(tiny_model.config._attn_implementation) == impl_before


def _hand_weights_kvzip() -> torch.Tensor:
    # 4 query heads in 2 groups of 2, 3 reconstruction rows, 5 key columns of
    # which the leading 3 hold the scored source positions.
    return torch.tensor(
        [
            [[0.1, 0.9, 0.2, 0.0, 0.0], [0.5, 0.1, 0.3, 0.0, 0.0], [0.2, 0.2, 0.2, 0.0, 0.0]],
            [[0.7, 0.0, 0.1, 0.0, 0.0], [0.0, 0.4, 0.8, 0.0, 0.0], [0.3, 0.3, 0.3, 0.0, 0.0]],
            [[0.0, 0.0, 0.6, 0.0, 0.0], [0.9, 0.1, 0.1, 0.0, 0.0], [0.4, 0.5, 0.0, 0.0, 0.0]],
            [[0.2, 0.8, 0.0, 0.0, 0.0], [0.1, 0.2, 0.4, 0.0, 0.0], [0.0, 0.0, 0.7, 0.0, 0.0]],
        ]
    )


def test_layer_stream_score_is_the_max_over_rows_and_grouped_query_heads() -> None:
    weights = _hand_weights_kvzip()

    scores = reconstruction_stream_scores(weights, source_length=3, kv_groups=2)

    manual = [
        [
            max(
                float(weights[head, row, column])
                for head in range(kv_head * 2, kv_head * 2 + 2)
                for row in range(3)
            )
            for column in range(3)
        ]
        for kv_head in range(2)
    ]
    assert scores.shape == (2, 3)
    assert scores.dtype == torch.float32
    assert scores.tolist() == manual
    # written out so the expected numbers are readable, not merely rederived
    torch.testing.assert_close(
        scores,
        torch.tensor([[0.7, 0.9, 0.8], [0.9, 0.8, 0.7]]),
        atol=1e-7,
        rtol=0,
    )


def test_layer_stream_score_rejects_geometry_it_cannot_group() -> None:
    weights = _hand_weights_kvzip()
    with pytest.raises(ValueError):
        reconstruction_stream_scores(weights, source_length=3, kv_groups=3)
    with pytest.raises(ValueError):
        reconstruction_stream_scores(weights, source_length=9, kv_groups=2)
    with pytest.raises(ValueError):
        reconstruction_stream_scores(weights[0], source_length=3, kv_groups=2)


def _upstream_get_score(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    *,
    sink: int,
    start_idx: int,
    end_idx: int,
) -> torch.Tensor:
    """A literal transcription of KVzip `attention/score.py::KVScore._get_score`.

    Public source pin: github.com/snu-mllab/KVzip at commit
    5d8472975d286e3da6b52178af39d18904330c0d.
    """
    bsz, _, q_len, head_dim = query_states.shape
    num_kv = key_states.size(1)

    query_states = query_states.view(bsz, num_kv, -1, q_len, head_dim)
    key_states = torch.cat(
        [
            key_states[:, :, :sink],
            key_states[:, :, start_idx:end_idx],
            key_states[:, :, -q_len:],
        ],
        dim=2,
    )
    key_states = key_states.unsqueeze(2).transpose(-2, -1).contiguous()
    ctx_len = end_idx - start_idx

    attn_weights = torch.matmul(query_states, key_states) / math.sqrt(head_dim)

    mask = torch.full((q_len, q_len), torch.finfo(attn_weights.dtype).min, dtype=attn_weights.dtype)
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    attn_weights[..., -q_len:, -q_len:] += mask[None, None, None, :, :]

    attn_weights = torch.softmax(attn_weights, dim=-1)
    attn_weights = attn_weights[..., sink : sink + ctx_len]
    return attn_weights.amax(dim=(-3, -2))


@pytest.mark.parametrize("sink", [0, 1])
def test_layer_score_matches_the_official_reconstruction_score(sink: int) -> None:
    torch.manual_seed(11)
    memory, rows, head_dim = 9, 4, 4
    query = torch.randn(1, 4, rows, head_dim, dtype=torch.float64)
    key = torch.randn(1, 2, memory + rows, head_dim, dtype=torch.float64)

    ours = native_reconstruction_scores(
        query,
        key,
        source_start=sink,
        source_end=memory,
        scaling=1 / math.sqrt(head_dim),
        sink_size=sink,
    )
    theirs = _upstream_get_score(query, key, sink=sink, start_idx=sink, end_idx=memory)

    assert ours.shape == theirs.shape == (1, 2, memory - sink)
    torch.testing.assert_close(ours, theirs, atol=0, rtol=0)


def test_pure_stream_reduction_agrees_with_the_native_layer_path() -> None:
    torch.manual_seed(12)
    memory, rows, head_dim = 7, 3, 4
    query = torch.randn(1, 6, rows, head_dim, dtype=torch.float64)
    key = torch.randn(1, 3, memory + rows, head_dim, dtype=torch.float64)

    native = native_reconstruction_scores(
        query,
        key,
        source_start=0,
        source_end=memory,
        scaling=1 / math.sqrt(head_dim),
    )

    keys = torch.repeat_interleave(key, 2, dim=1)
    logits = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(head_dim)
    index = torch.arange(rows)
    logits[..., -rows:] += torch.where(
        index.unsqueeze(0) <= index.unsqueeze(1),
        torch.zeros((), dtype=logits.dtype),
        torch.full((), torch.finfo(logits.dtype).min, dtype=logits.dtype),
    )
    weights = torch.softmax(logits, dim=-1)[0]
    pure = reconstruction_stream_scores(weights, source_length=memory, kv_groups=2)

    torch.testing.assert_close(native[0].to(torch.float32), pure, atol=1e-6, rtol=0)


def test_projection_is_invariant_to_uniform_layer_or_kv_head_duplication() -> None:
    torch.manual_seed(13)
    native = torch.rand(3, 2, 16)
    reference = project_position_scores(native)

    layer_doubled = project_position_scores(torch.cat([native, native], dim=0))
    head_doubled = project_position_scores(torch.cat([native, native], dim=1))

    torch.testing.assert_close(layer_doubled, reference, atol=1e-6, rtol=0)
    torch.testing.assert_close(head_doubled, reference, atol=1e-6, rtol=0)
    assert torch.equal(layer_doubled.argsort(), reference.argsort())
    assert torch.equal(head_doubled.argsort(), reference.argsort())


def test_mid_chunk_failure_restores_cache_and_surfaces_the_original_error(
    tiny_model, monkeypatch
) -> None:
    context = random_ids(length=12, seed=914)
    with torch.no_grad():
        past = tiny_model(input_ids=context, use_cache=True).past_key_values
    snapshot = _cache_snapshot(past)
    impl_before = str(tiny_model.config._attn_implementation)
    original_forward = tiny_model.model.forward
    calls = {"n": 0}

    def failing_forward(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("injected chunk failure")
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(tiny_model.model, "forward", failing_forward)

    with pytest.raises(RuntimeError, match="injected chunk failure"):
        memory_kvzip_scores(tiny_model, past, context[:, :10], _Tokenizer(), chunk_size=4)

    assert calls["n"] >= 2
    assert cache_length(past) == 12
    _assert_cache_bytes_equal(past, snapshot)
    assert str(tiny_model.config._attn_implementation) == impl_before


def test_restore_failure_never_masks_the_original_error(tiny_model, monkeypatch) -> None:
    import rcc.transforms.select.baselines.kvzip as module

    context = random_ids(length=12, seed=915)
    with torch.no_grad():
        past = tiny_model(input_ids=context, use_cache=True).past_key_values

    def failing_forward(*args, **kwargs):
        raise ValueError("injected body failure")

    def failing_restore(*args, **kwargs):
        raise RuntimeError("injected restore failure")

    monkeypatch.setattr(tiny_model.model, "forward", failing_forward)
    monkeypatch.setattr(module, "_restore_cache", failing_restore)

    # The restore failure is terminal because a half-restored cache is unsafe
    # to use, while the original scoring failure remains its explicit cause.
    with pytest.raises(RuntimeError, match="injected restore failure") as caught:
        memory_kvzip_scores(tiny_model, past, context[:, :10], _Tokenizer(), chunk_size=4)

    visible: list[str] = []
    error: BaseException | None = caught.value
    while error is not None:
        visible.append(str(error))
        visible.extend(getattr(error, "__notes__", []))
        error = error.__cause__ or error.__context__
    combined = " | ".join(visible)
    assert "injected body failure" in combined
    assert "injected restore failure" in combined
