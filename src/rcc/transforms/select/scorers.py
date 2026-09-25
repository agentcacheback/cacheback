"""Attention-vote scorers for Select on the length axis, plus a gradient scorer.

Each vote scorer runs one prefill with `output_attentions=True`, so the model must be loaded
with eager attention. See docs/selectors.md, Attention-vote and gradient scorers.
"""

from __future__ import annotations

import warnings
from typing import Any, cast

import torch
import torch.nn.functional as F  # noqa: N812  canonical PyTorch alias

from rcc.cache import KVCache
from rcc.harness.handoff import handoff


def _batched(ids: torch.Tensor) -> torch.Tensor:
    """Normalize token ids to [1, n]; every scorer accepts [n] or [1, n]."""
    return ids.unsqueeze(0) if ids.ndim == 1 else ids


@torch.no_grad()
def _prefill_attentions(model: Any, context_ids: torch.Tensor) -> tuple[list[torch.Tensor], int]:
    """Prefill the context with output_attentions and return per-layer maps plus length."""
    context_ids = _batched(context_ids)
    length = int(context_ids.shape[1])
    position_ids = torch.arange(length, device=context_ids.device).unsqueeze(0)
    out = model(
        input_ids=context_ids,
        position_ids=position_ids,
        use_cache=False,
        return_dict=True,
        output_attentions=True,
    )
    if out.attentions is None:
        raise ValueError(
            "scorer got no attention weights: load the model with "
            'attn_implementation="eager" (SDPA and FlashAttention return None).'
        )
    return list(out.attentions), length


def _pool_cluster(score: torch.Tensor, kernel: int, mode: str = "max") -> torch.Tensor:
    """Apply SnapKV 1-D pooling-clustering, pulling contiguous neighbours up.

    A kernel of 1 or less is a no-op; the kernel must be odd so that the pooled
    vector stays aligned with its tokens.
    """
    if kernel <= 1:
        return score
    if kernel % 2 == 0:
        raise ValueError(
            f"pool kernel must be odd, got {kernel}: an even kernel with padding "
            "kernel // 2 yields one extra output and the trailing slice shifts "
            "every score half a position off its token"
        )
    x = score.view(1, 1, -1)
    pad = kernel // 2
    if mode == "avg":
        pooled = F.avg_pool1d(x, kernel_size=kernel, padding=pad, stride=1)
    else:
        pooled = F.max_pool1d(x, kernel_size=kernel, padding=pad, stride=1)
    return pooled.view(-1)[: score.shape[0]]


def h2o_scores(model: Any, context_ids: torch.Tensor) -> torch.Tensor:
    """H2O heavy-hitter score [n_ctx]: total attention mass each key receives from every row."""
    attentions, n_ctx = _prefill_attentions(model, context_ids)
    score = torch.zeros(n_ctx)
    for layer_attn in attentions:
        heads_mean = layer_attn[0].mean(dim=0)
        score += heads_mean.sum(dim=0).float().cpu()
    return score


def snapkv_scores(
    model: Any,
    context_ids: torch.Tensor,
    obs: int = 8,
    pool_kernel: int = 7,
    pool_mode: str = "max",
) -> torch.Tensor:
    """SnapKV score [n_ctx]: attention from the last `obs` rows, then 1-D pooling-clustering."""
    attentions, n_ctx = _prefill_attentions(model, context_ids)
    obs = max(1, min(obs, n_ctx))
    score = torch.zeros(n_ctx)
    for layer_attn in attentions:
        heads_mean = layer_attn[0].mean(dim=0)
        score += heads_mean[-obs:, :].sum(dim=0).float().cpu()
    return _pool_cluster(score, pool_kernel, pool_mode)


@torch.no_grad()
def _joint_attentions(
    model: Any, context_ids: torch.Tensor, tail_ids: torch.Tensor
) -> tuple[list[torch.Tensor], int, int]:
    """One eager forward over [context | tail]; returns per-layer maps, n_ctx, and total len."""
    context_ids = _batched(context_ids)
    tail_ids = _batched(tail_ids)
    n_ctx = int(context_ids.shape[1])
    joint = torch.cat([context_ids, tail_ids], dim=1)
    total = int(joint.shape[1])
    position_ids = torch.arange(total, device=joint.device).unsqueeze(0)
    out = model(
        input_ids=joint,
        position_ids=position_ids,
        use_cache=False,
        return_dict=True,
        output_attentions=True,
    )
    if out.attentions is None:
        raise ValueError(
            "scorer got no attention weights: load the model with "
            'attn_implementation="eager" (SDPA and FlashAttention return None).'
        )
    return list(out.attentions), n_ctx, total


def _vote_over_context(
    attentions: list[torch.Tensor], row_lo: int, row_hi: int, n_cols: int
) -> torch.Tensor:
    """Sum attention from rows [row_lo, row_hi) onto columns [0, n_cols), over layers/heads."""
    score = torch.zeros(n_cols)
    for layer_attn in attentions:
        heads_mean = layer_attn[0].mean(dim=0)
        score += heads_mean[row_lo:row_hi, :n_cols].sum(dim=0).float().cpu()
    return score


def _find_subsequence(haystack: list[int], needle: list[int]) -> int:
    """Return the start index of the first exact contiguous match, or -1 if absent."""
    span = len(needle)
    if span == 0:
        return -1
    for i in range(len(haystack) - span + 1):
        if haystack[i : i + span] == needle:
            return i
    return -1


def snapkv_query_scores(
    model: Any,
    context_ids: torch.Tensor,
    prompt_ids: torch.Tensor,
    question_ids: torch.Tensor,
    *,
    pool_kernel: int = 7,
    pool_mode: str = "max",
) -> torch.Tensor:
    """Query-aware score [n_ctx]: votes from the bare-question rows over the context.

    One eager forward over [context | prompt] sums the attention the question's own rows
    send onto the context columns. See docs/selectors.md, Query-aware attention votes.

    Args:
        model: A causal LM loaded with eager attention.
        context_ids: Context token ids, shape [n_ctx] or [1, n_ctx].
        prompt_ids: The full decode prompt token ids that contain the question.
        question_ids: The bare question token ids to locate inside `prompt_ids`.
        pool_kernel: 1-D pooling-clustering kernel (<= 1 disables pooling).
        pool_mode: "max" (default) or "avg" pooling.
    """
    prompt_row = prompt_ids[0] if prompt_ids.ndim == 2 else prompt_ids
    question_row = question_ids[0] if question_ids.ndim == 2 else question_ids
    prompt_list = [int(t) for t in cast(Any, prompt_row).tolist()]
    question_list = [int(t) for t in cast(Any, question_row).tolist()]
    off = _find_subsequence(prompt_list, question_list)
    q_len = len(question_list)
    if off < 0:
        # The word-initial first token usually differs between the standalone
        # tokenization and the in-template one (leading-space BPE). An empty
        # interior returns -1, so short questions need no special case.
        interior = question_list[1:-1]
        interior_off = _find_subsequence(prompt_list, interior)
        if interior_off >= 0:
            off, q_len = interior_off, len(interior)
    attentions, n_ctx, total = _joint_attentions(model, context_ids, prompt_ids)
    if off >= 0:
        row_lo, row_hi = n_ctx + off, n_ctx + off + q_len
    else:
        row_lo, row_hi = n_ctx, total
    score = _vote_over_context(attentions, row_lo, row_hi, n_ctx)
    return _pool_cluster(score, pool_kernel, pool_mode)


@torch.no_grad()
def _generate_probe_questions(
    model: Any, tokenizer: Any, context_text: str, n_questions: int, max_new_tokens: int
) -> str:
    """Greedy-generate short factual probe questions the context is asked to answer."""
    content = (
        "Here is a document:\n\n"
        f"{context_text}\n\n"
        f"Write {n_questions} short factual questions that this document answers. "
        "Reply with one question per line, nothing else."
    )
    messages = [{"role": "user", "content": content}]
    prompt: str | None = None
    for extra in ({"enable_thinking": False}, {}):
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **extra
            )
            break
        except (TypeError, ValueError):
            continue
    if prompt is None:
        prompt = content
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = encoded["input_ids"].to(next(model.parameters()).device)
    generated = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(generated[0, input_ids.shape[1] :], skip_special_tokens=True)
    return text.strip()


@torch.no_grad()
def selfq_scores(
    model: Any,
    tokenizer: Any,
    context_ids: torch.Tensor,
    *,
    n_questions: int = 5,
    max_new_tokens: int = 96,
    pool_kernel: int = 7,
    pool_mode: str = "max",
) -> torch.Tensor:
    """Self-interrogation score [n_ctx]: the model's own probe questions vote over the context.

    Query-agnostic scorer for the store setting; see docs/selectors.md, Self-interrogation votes.

    Args:
        model: A causal LM loaded with eager attention.
        tokenizer: Renders the context to text and tokenizes the generated questions.
        context_ids: Context token ids, shape [n_ctx] or [1, n_ctx].
        n_questions: Number of probe questions to request.
        max_new_tokens: Greedy generation budget for the probe questions.
        pool_kernel: 1-D pooling-clustering kernel (<= 1 disables pooling).
        pool_mode: "max" (default) or "avg" pooling.
    """
    ctx = _batched(context_ids)
    context_text = tokenizer.decode(ctx[0], skip_special_tokens=True)
    question_text = _generate_probe_questions(
        model, tokenizer, context_text, n_questions, max_new_tokens
    )
    probe = tokenizer(
        question_text or "What is this document about?",
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=256,
    )
    probe_ids = probe["input_ids"].to(ctx.device)
    attentions, n_ctx, total = _joint_attentions(model, ctx, probe_ids)
    score = _vote_over_context(attentions, n_ctx, total, n_ctx)
    return _pool_cluster(score, pool_kernel, pool_mode)


def grad_scores(
    model: Any,
    context_ids: torch.Tensor,
    prompt_ids: torch.Tensor,
    *,
    draft_tokens: int = 16,
    pool_kernel: int = 7,
    pool_mode: str = "max",
) -> torch.Tensor:
    """Gradient-attribution score [n_ctx]: how much each cached column moves the draft answer.

    Gradient times activation over the cached keys and values; see docs/selectors.md, Gradient
    attribution. fp16 can underflow to an all-zero score and a warning; bf16 and fp32 are safe.

    Args:
        model: A causal LM under any attention implementation (no eager requirement).
        context_ids: Context token ids, shape [n_ctx] or [1, n_ctx].
        prompt_ids: The decode prompt token ids the answer draft is conditioned on.
        draft_tokens: Greedy budget for the answer draft that is attributed.
        pool_kernel: 1-D pooling-clustering kernel (<= 1 disables pooling).
        pool_mode: "max" (default) or "avg" pooling.
    """
    if draft_tokens < 1:
        raise ValueError(f"draft_tokens must be positive, got {draft_tokens}")
    prompt = _batched(prompt_ids)
    cache = KVCache.from_prefill(model, context_ids)
    n_ctx = cache.length
    draft = handoff(cache, model, prompt_ids, max_new_tokens=draft_tokens, stop_at_eos=True)
    if int(draft.shape[1]) == 0:
        # There is nothing to attribute and no substitute draft to fall back on.
        raise ValueError(
            "empty answer draft: the model emitted end-of-turn immediately, "
            "so there is nothing to attribute; rework the prompt"
        )
    q, k = int(prompt.shape[1]), int(draft.shape[1])
    with torch.enable_grad():
        # detach() shares storage with the prefill cache, which nothing writes in
        # place, so the scorer holds one cache copy rather than two.
        keys_leaf = cache.keys.detach().requires_grad_(True)
        values_leaf = cache.values.detach().requires_grad_(True)
        past = KVCache(
            keys=keys_leaf, values=values_leaf, positions=cache.positions, rope=cache.rope
        ).to_hf_cache()
        input_ids = torch.cat([prompt, draft], dim=1)
        device = input_ids.device
        out = model(
            input_ids=input_ids,
            attention_mask=torch.ones(1, n_ctx + q + k, dtype=torch.long, device=device),
            past_key_values=past,
            position_ids=torch.arange(n_ctx, n_ctx + q + k, device=device).unsqueeze(0),
            use_cache=True,
            return_dict=True,
        )
        log_probs = torch.log_softmax(out.logits[0, q - 1 : q + k - 1, :].float(), dim=-1)
        rows = torch.arange(k, device=log_probs.device)
        loss = log_probs[rows, draft[0].to(log_probs.device)].sum()
        grads = torch.autograd.grad(loss, [keys_leaf, values_leaf])
    gk, gv = grads[0], grads[1]
    with torch.no_grad():
        score = ((gk * keys_leaf).sum(-1).abs() + (gv * values_leaf).sum(-1).abs()).sum(dim=(0, 1))
    score = score.float().cpu()
    if not bool(score.any()):
        warnings.warn(
            "grad_scores returned all-zero scores (fp16 gradient underflow is the "
            "usual cause); Select will break ties arbitrarily",
            stacklevel=2,
        )
    return _pool_cluster(score, pool_kernel, pool_mode)
