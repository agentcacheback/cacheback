# Benchmarks

## FanOutQA, natural fifty-question panel

`FANOUTQA_NATURAL_DEV50` is fifty official FanOutQA dev questions, chosen as
the first qualifying ids in ascending sha256 of the question id from a pool
that excludes the thirty selector-development questions. A question
qualifies when every one of its three workers can carry at least 40,000
Qwen3-8B tokens of real page text. An id whose trim removed a locatable gold
leaf is skipped, and the next qualifying id takes its place.

Each worker holds a character span of the cleaned official Wikipedia page
text, cut by the `max_min_query` allocation policy toward 50,000 content
tokens per worker, with no filler and no fixed prompt width. The panel ships
as one sealed bundle (`rcc.data fetch fanoutqa-natural-dev50`) that carries
the worker texts, the page provenance, the cleaned pages every span was cut
from, the pinned dev question index, and per-file digests. Every family
tokenizes the same bytes with its own tokenizer at prepare time and never
re-cuts a page: worker prompts run 40,015 to 49,995 tokens on Qwen and up to
53,104 on the widest of the other families. The
registered ceiling of 53,248 is a ceiling and not a width: the receiver's window
arithmetic reserves against it, but nothing renders to it.

Every item runs the same eleven arms: the question-only floor, three text
report arms (the receiver's own size, a medium sender, a small sender), and
the seven CacheBack ratios 2 through 128 on 16-token spans. Gemma and
Ministral add a `full` control that hands over the whole worker cache. Each
arm draws three receiver samples per item with seeds derived from the panel's
registered namespace, the question id, the arm, and the sample tag, so a rerun
reproduces the draws' seeds exactly.

Scoring is `fanoutqa-equivalence-v1`: the registered string proxy over the
question's gold leaves, with nine typo and page-contradicted reference
corrections (`gold_patch.json`), integer-valued decimals treated as equal
(`72`, `72.0`), square-unit spellings treated as equal (`km2`, `km²`), and a
short audited alias list for two-person references. Strict accuracy requires
every reference group; loose accuracy is the fraction of groups found.

### The frozen census and the shard split

Page ownership on every FanOutQA panel comes from a frozen census rather than
from the live pages. `fanoutqa_freeze.json` fixes the shard size S, the 208-item
eligible set, the 30/178 dev and held-out split, and every page's census token
count. Eligibility is never re-derived, because one eligible item clears the 2S
line by 0.148 percent.

Pages are assigned whole, longest first, into the lighter shard, with ties and
equal loads broken by sha256, and a shard already holding
`ceil(content pages / workers)` pages takes no more. Zero-token pages are
excluded up front: an empty pinned revision is not a page.

Three registered policies then cut the assigned pages to the budget.
`longest_tail_trim`, the evidence-provided loader's own policy, trims tail-first
from the longest page of an overfull shard until it fits, never below one token
per page. `max_min_prefix` reserves every joiner token and water-fills the page
prefixes, so short pages saturate at full length and the rest receive one common
integer prefix length, with at most one token of deterministic remainder skew.
`max_min_query`, the natural panel's policy, allocates on top of
`max_min_prefix` and is described next.

### Query-coverage spans

Under `max_min_query` a worker's quota is filled with the first half of each
shortened page followed by that page's 256-token chunks, ranked against the
main question on lexical and numeric overlap with deterministic ties. The
ranking reads the question alone, never a sub-answer or an answer value. Table
rows and their headers cross the 256-token boundary, so a chunk is ranked over
its one-chunk neighborhood, while table structure is counted only in the bytes
that would be retained and no neighbor is selected automatically.

Separator demand is monotone between page-saturation points but drops when a
page becomes complete, because a full article is one contiguous segment, so the
budget search walks each monotone interval and a feasible full page cannot hide
behind an infeasible almost-full allocation.

### The natural bundle

A profile pins the bundle three ways: by the manifest digest, by the sha256 of
`panel.json`, and by the pinned question index. The archive digest is checked
only when the archive sits beside the tree. The layout of
`fanoutqa-natural-<panel>-source-v1` is:

    <root>/manifest.json         file roster with a sha256 per file
    <root>/panel.json            dataset pins, policy, per-item summary
    <root>/items/<qid>.json      worker texts, page provenance, spans
    <root>/source_cache/fanout-final-dev.json   the pinned question index

### The latent cut

`fanoutqa-natural-dev50-latent` is the same panel registered with only the
seven support-latent arms. It shares the bundle, the fifty items, the config
fingerprint, the seeds, and the decode; it owns its profile id, its prepared
subtree and artifact digest (the prepared items carry the roster), and its
serving registration digest. It exists for ablations of the latent channel
alone, where the floor and text arms would only repeat the eleven-arm bank.

## LongBench v2, chain of four hops

`LONGBENCH_COA_EASY50_*` is fifty easy rows of LongBench v2 (dataset revision
pinned in `panel.py`), drawn one question per context by a salted hash with
Hamilton quotas over domain and difficulty inside three context-length
strata (100K to 150K, 150K to 200K, 200K to 250K tokens) of the 503 official
rows, with 25, 16, and 9 taken from the three strata. Where several rows share
one context, only the row with the lowest hash survives. The selection reads
only the context length, the domain, the difficulty, and the id. The build
CLI's `fetch-raw` downloads the raw file and verifies its digest,
`build-bundle` builds the source bundle the entry consumes, and `select`
re-derives the sealed panel and checks it against the frozen digests. The
entry binds a bundle by its manifest digest and logical fingerprint, which
cover every file's bytes; the compressed archive's digest is recorded but not
compared, since two zstd builds compress the same tar differently. Every
context is cut at whole-token quartiles into four chunks whose concatenation
is the source byte for byte; the chunk ledger and its digests are in the tree.

One producer seat walks the four chunks in order. After every hop the whole
sequence (carried rows, the new chunk, the wrapper, the forty latent rows) is
re-scored by CacheBack and cut under one budget law, with nothing carried
protected except the sink and the current latent rows. The re-ranked handoff
law keeps `carried + ceil(new_rows / ratio)` rows, so the memory grows by one
compressed chunk per hop; the bounded law keeps `min(budget_rows, rows)`, one
fixed total across the whole chain. The text arms rewrite a note at every hop
under an 8,000-word brief, drawn at presence penalty 1.5 and a 16,000-token
ceiling; the floor answers from the question and choices alone. The receiver
answers a four-way multiple choice from the terminal rows; the score is exact
letter accuracy.

### Chunk boundaries

The boundaries sit at `floor(t S / 4)` whole-context tokens for `t` in one to
three, mapped to the character where token `b_t - 1` ends. The four substrings
concatenate back to the source byte for byte, are retokenized on their own, and
each must stay under the source chunk ceiling, with nothing padded, snapped,
overlapped, repeated, or truncated.

The guarantee is over the raw substrings, not over a decode of the token ids:
the pinned tokenizer normalizes to NFC and a few panel contexts are not NFC
stable, so a text arm, which decodes one chunk back for its sender, reads the
normalized body on those items. Every hop records the `chunk_text_sha256` and
`chunk_text_tokens` of the body it read, so the difference is visible.

### Arm rosters

The three config keys share one panel object, the chunks, the prompts, the
scorer, and the seed namespace, so every row pairs item for item with its mate
in the other banks. The re-ranked ladder runs ratios 4 through 128: ratio 2 is
absent because at that count hop four of the nine longest easy items does not
fit the window. The bounded ladder is powers of two from 65,536, the largest
fixed memory the served window admits beside one raw chunk (61,466 rows on this
tokenizer), down to 2,048; on this panel the cap binds from hop two on every
item. The Nemotron bounded ladder starts at 32,768: on its tokenizer the
65,536 rung does not fit every item's worst hop inside the capture window.
The text key carries no latent ladder, only the floor and the three text
arms, and its note draw moves two values, presence penalty 1.5 and a
16,000-token note ceiling, because a draw with no penalty can fall into a
repeat cycle and ignore the brief's word cap. The receiver's answer decode is
unchanged, so a text row stays comparable to a latent row and to the floor.
