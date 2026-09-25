# Selectors

Every selector in `rcc.transforms.select` scores the positions of one worker's
KV cache and keeps a budgeted subset. `Select(scores, ratio=r)` keeps one
position in `r`; `Select(scores, budget=n)` keeps `n`; a `schedule` turns the
position-wise keep into a span-wise keep. Kept positions are RoPE-relocated
to a contiguous prefix before the receiver reads them, and a keep set that is
already contiguous from 0 is returned without rotation, bit-identical to the
full KV.

## The span keep law

`span_keep` tiles the score vector into near-equal contiguous spans, ranks them
by mean token score, and adds whole spans until the budget is spent. The span
that overshoots the budget contributes exactly the room that is left, as one
contiguous window around its peak token, clamped inside the span and widened
past that width wherever it covers positions already kept, so a span never
scatters into isolated tokens. The sink is force-kept and counted inside the
budget, and the keep list has length exactly `min(budget, n)`. A kept span is a
contiguous run relocated to a contiguous target run, so its intra-span offsets
survive the RoPE relocation and exact-copy content that a token-wise top-k
would fragment is preserved.

## CacheBack (`query_support`)

CacheBack is receiver-conditioned: the score of a worker position is the
attention it receives from the receiver's query rows, read off the worker's
own cache during one extra forward pass over the query. Three vectors come
out of that capture (`memory_votes_with_energies`):

- `snap`: mean query attention, the pooled attention mass each position receives
  from the query rows, max-pooled over a 7-token kernel;
- `energy`: the attention from the question rows, averaged over the query
  heads of each KV head, pooled over the same kernel, then squared, so it is
  the order-2 moment of the attention density each position draws;
- `row_energy`: the same quantity computed row by row for each question
  token before averaging, so a position a single question row leans on keeps
  that concentration instead of having it averaged away.

The composed score (`support_corrected_scores`) multiplies `snap` by the
support ratio `(row_energy / energy) ** alpha`, computed in float32 with that
dtype's smallest normal value as its floor. The three vectors must describe the
same memory positions. The registered point is order `p = 2` and correction strength
`alpha = 2` (`support-p2-a2`), chosen on a 30-item development sweep separate
from every evaluation panel. Positions whose energies are both zero receive the
neutral correction of one, which is the all-zero limit, so no epsilon is needed.

Scores are then pooled over fixed 16-token spans (`fixed_spans`), and spans
are kept whole. Position zero (the attention sink) and the forty latent rows
the worker rolled after its prompt are always kept.

### Stream weights

The released Gemma configurations score positions using global-attention
layers only. The library also supports variants that include sliding layers;
the weighting and folding rules below describe those additional variants.

A layer's support moments are sums over its key-value streams, and a hybrid
model does not give every layer the same number of them: at the pinned Gemma
revision the sliding layers carry eight and the global layers one. Adding the
layers up untouched would therefore weight each one by its stream count instead
of counting it once, so every layer is rescaled to the largest stream
population before the sum. On a model whose layers all carry the same number,
every weight is one and the arithmetic is unchanged to the bit. The
order-infinity moments are maxima over streams rather than sums, so they are
left alone.

### Sliding-layer folds

At the pinned Gemma revision 8 of the 48 layers attend to the whole context and
the other 40 attend only inside a 1024-token window. Both banked variants count
the votes those 40 layers cast, and they differ only in how those votes are
added up. The tail variant (`slidetail`) sums them, so a position near the end
of the context can collect votes from all 48 layers while a position deep in it
collects 8. The normalised variant (`slidenorm`) averages over the layers that
actually cover each position, which divides that coverage back out; it is the
control for recency bias, and a position inside one sliding window is divided
by 9, that layer plus the 8 global ones, not by 48.

Dividing every moment by the same per-position coverage leaves the support
correction unchanged, since it is a ratio of two moments that both scale, and
rescales the attention-mass vote alone. The division has to happen on the raw
per-position sums, before they are pooled, because a maximum pool and a
division do not commute at a window boundary: pooling first would carry a
neighbour's larger raw sum into a position and then divide it by that
position's own smaller coverage. The moments are divided as well, and for them
the order does not matter, because they are pooled when they are captured and
only summed here.

## Baselines

`baselines/` holds ports of the established cache-eviction scorers the paper
compares against on the same span schedule and budgets: H2O (accumulated
attention mass), StreamingLLM (recency), ChunkKV, and KVzip; SnapKV, with and
without the query rows, is in `scorers.py`. Mean query attention is CacheBack
without the support correction. Internal keys such as `snap` and `qsnap` retain
their original names for compatibility with recorded data.
Each baseline is scored by the same fixed-span keep law so the comparison
changes only the score.

## Attention-vote and gradient scorers

`scorers.py` holds three scorers that run outside the CacheBack capture and
score a context on their own. The two attention-vote scorers sum the attention
each context position receives over the layers and average it over the heads,
giving one score vector for the whole context; because they read the model's
attention maps, the model has to be loaded with eager attention, since the
faster attention kernels return no maps at all. `grad_scores` reads gradients
instead and runs under any attention implementation.

### Query-aware attention votes

`snapkv_query_scores` locates the question as an exact contiguous subsequence
of the decode prompt so that only the bare-question rows vote: the chat
template boilerplate around the question votes generically and dilutes the
context signal. If the exact match fails and the question has more than two
tokens, the search retries with the interior tokens `question_ids[1:-1]`,
since the template boundary can absorb the first and last question token. If
that also fails, all prompt rows vote.

### Self-interrogation votes

`selfq_scores` is the query-agnostic scorer for the store setting, where there
is no user query at encode time. The model greedy-generates `n_questions`
short factual questions the context answers, and one eager forward over
`[context | questions]` sums the attention those probe rows send onto the
context columns. When generation yields nothing, a single generic probe stands
in so the forward always has voting rows.

### Gradient attribution

`grad_scores` is a first-order attribution of a query-conditioned answer
log-likelihood to each cached column. The context is prefilled once, the model
greedy-decodes a short answer draft from the prompt, and one backward pass
over the teacher-forced draft log-likelihood gives the gradient with respect
to the cached keys and values. The per-column score is gradient times
activation, the first-order change in the draft likelihood from zeroing that
column, summed over layers and heads, keys plus values. In fp16 the gradients
can underflow to an all-zero score and a warning; bf16 and fp32 are safe.
