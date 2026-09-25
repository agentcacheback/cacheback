# Running a benchmark

One node with eight 80 GB H100 GPUs runs one config end to end:

```bash
python -m rcc.data fetch fanoutqa-natural-dev50
python -m rcc.run.entry --config configs/qwen-fanoutqa-natural-dev50.toml \
    --bundle data/fanoutqa-natural-dev50
```

`rcc.run.entry` resolves the config into an immutable plan (the registered
model, benchmark, topology, arm roster, item range, decode settings, and
seeds), downloads the pinned tokenizer files, builds the prepared panel with
the family's tokenizer, records the placement table, and then drives every
arm in order through the split-fleet driver. It ends by publishing
`report.json` under the plan's output directory (`runs/<benchmark>/<run id>/`
by default). `--arm` restricts a run to named arms. The Qwen 3 and Nemotron
rosters go by policy name (`none`, `qwen3_8b_r8_w16_support`), Gemma and
Ministral by semantic name (`issue_only`, `latent_query_support_r8`); the plan
the `plan` command writes lists both for every arm. The Gemma configs run their
roster whole, in registered order, and refuse a subset.

The optional `[ablation]` table names the one registered ablation,
`judger_question`. It replaces the question the selector's judger carries when
it scores a rolled worker cache; the worker prompts and the final receiver
prompt keep the item's own question. The value enters the plan's scientific
identity, reaches every seat as `RCC_FANOUT_JUDGER_QUESTION` (set in the entry
process and inherited by the split driver's spawned seats, like the benchmark
name), and is recorded in every banked row. Only the FanOutQA route lanes,
whose shared producer renders the judger, may name it; the resident Gemma and
Ministral lanes and the chain topology refuse it. `configs/qwen-fanoutqa-natural-dev50-judger-ablation.toml`
runs it over the eleven arms, and the `-latent-` config over the seven latent
arms alone (`docs/benchmarks.md`, the latent cut).

The selector-development experiment of Section 3.2 and Appendix D is a
separate capture-only sweep with its own bundle and its own three CPU commands:
`docs/selector-dev.md`.

## The split fleet

An arm owns the whole node for its duration. The driver spawns one process per
GPU seat; the placement table under `rcc.hardware.placements` says how many
seats produce and how many receive for each arm. A producer seat serves the
worker model: it prefills a worker's context, rolls the forty latent steps,
captures the CacheBack scores, and publishes the selected handoff to a
filesystem queue. A receiver seat serves the receiver model: it claims a
published handoff, assembles the receiver prompt around it, and decodes the
three samples. Text arms replace the latent producer with a sender that
writes the worker report; the floor and the fused text arm run receivers
only.

Every seat appends its rows to its own durable bank
(`arms/<arm>/workers/gpu<n>/raw.jsonl`) with an fsync per row, and completed
items are skipped when an attempt is relaunched under a fresh attempt id.
Stage rows carry the clocks the paper reports: the fleet clock runs from the
batch's arrival to the receiver's first sample end, so the headline latency
is set by the producer queue on a fixed node, and the per-item compute times
are banked beside it.

`split_driver report` merges the banks into `report.json`: per-arm accuracy
with paired bootstrap intervals, the clocks, and the shipped payload bytes.

## Reading a run back

`rcc.run.cli` takes the config, a `--plan-out` path, and a command: `plan`
writes the resolved plan and stops, and three commands read a run back.

```bash
python -m rcc.run.cli rescore-results --config C --plan-out P \
    --results runs/<benchmark>/<run id>/report/selected_results.jsonl \
    --runtime-bank runs/<benchmark>/<run id> --rescore-out rescore.json
python -m rcc.run.cli score-results --config C --plan-out P --results R \
    --source-bundle data/fanoutqa-natural-dev50 --references original --out scores.json
python -m rcc.run.cli handoff-recall --config C --plan-out P \
    --run-root runs/<benchmark>/<run id> --source-bundle data/fanoutqa-natural-dev50 \
    --out recall.json
```

`rescore-results` re-reads the banked raw tokens with the family's own reader
and refuses a row it does not reproduce. `score-results` scores the banked
answers under a named reference set: `fanoutqa-equivalence-v1` is what the
run scored itself with; `original` is the dataset's references as shipped,
before the nine corrections and the equivalent forms, which is the comparison
the paper's sensitivity table makes. `handoff-recall` scores what each handoff
carried against the same references: the three banked reports of a text arm,
or the source tokens a latent arm kept, decoded one contiguous run at a time so
a reference string never forms across a dropped position. Generated latent
rows carry no token and are left out; each item and arm counts once, not once
per receiver draw; the `source` entry is the coverage of the whole worker
prompts before selection. It reads the banks and, for Gemma and Ministral,
the selection files their producers publish beside the handoffs, and checks
every keep set against the selection hash the receiver banked. Ministral also
takes `--tokenizer-snapshot`. The breakdowns by question-given and
source-locatable leaves are not part of the command.

### Fleet latency

A merged arm carries two clocks per item. The compute-scope number the family's
own engine measured is retained, and the published time to first token and time
to end of answer are the fleet numbers the stage ledger measured, which also
count the wait in the producer queue, the handoff, and contention on the
receiver. An arm is priced on what the caller waited for rather than on what one
engine was busy with, and the span out to the last of the three samples is
published beside the headline. An arm whose receiver reads no handoff has no
producer, handoff or receiver-queue span at all: those fields are present and
empty rather than zero, so an average over them skips the empty ones.

### Ministral producer memory

The capture producer runs a large side pass outside the vLLM allocator, in the
same process and on the same device: the dense HF cache rebuilt on the GPU, the
capture forward's copy of it, the repeat_kv expansion, the per-chunk fp32
softmax, and the page-extraction transient. Together they come to about
18.7 GiB at the registered 49,999-token worker prompt.

The floor on `gpu_memory_utilization` is 0.695, below which one
`max_model_len=180224` request cannot be served: it needs 27.50 GiB of pool and
vLLM refuses at warm. The registered 0.72 buys a 29.49 GiB pool, about 193K
tokens, and leaves 21.4 GiB of side headroom.

## Trying the path without a GPU

`tests/test_entry.py` is the runnable example of everything above on CPU. It
writes a one-question natural bundle, points the registered profile at it,
replaces the tokenizer download and the resident engine with fakes, runs the
seats as threads, and drives `rcc.run.entry` through the same prepare, drive,
merge and report steps a node runs, in about three seconds.

## Environments

The paper's runs used two serving stacks, pinned in `requirements/`: Qwen 3
on vLLM 0.11.1, transformers 4.57.1, and torch 2.9; Gemma 4, Ministral 3, and
Nemotron Nano 2 on vLLM 0.26.0, transformers 5.13.0, and torch 2.11. Install
one file, then `pip install -e . --no-deps`. The registration module of each
family under `src/rcc/models/<family>/` pins the versions and checkpoint
revisions, and the entry downloads the family's checkpoints before the first
arm. Gemma 4 and Ministral 3 are gated on Hugging Face and need a logged-in
account that has accepted their terms.

## Report ceiling

A rescore holds every banked worker report to the registered ceiling, measuring
the content left after any stop marker is trimmed rather than counting the raw
ids. Because the draw was capped at the ceiling, how a report ended and how long
it is say the same thing: a report that ended because it ran out of room fills
the ceiling exactly, and one that stopped on its own spent a token on the stop
marker and so comes in at least one token short.

A worker whose closing was injected banks its head, its closer, and the
continuation as a single draw, recorded under the head's ending, so it is
allowed that same bound again plus the closer and the lane's closing budget. A
lane that injects nothing is held to the plain ceiling.

## Gemma embedding tolerance

One of the build checks compares the token rows the payload ships against what
the model's own embedding layer produces for the same tokens. Both run the same
operations on the same device, so they normally agree bit for bit, and the
record says so.

A checkpoint can still round them apart, because Gemma scales its embeddings by
a constant that is stored at the weights' own precision while the reference
computes that constant at full precision. The gap that allows is one step of the
shipped bfloat16 format, which at the magnitudes involved is 0.125, or about
0.78 percent.

That bound is where the tolerance comes from; it is not a fitted number. A real
mistake, a missing scale, a row off by one, a transposed table, or the wrong
precision, moves rows by whole percents or changes their shape, so it is not
something the tolerance can hide.

## Ministral thought-block reading

Ministral wraps its reasoning in a thought block, and its tokenizer can write
the opening and closing markers either as dedicated tokens or as ordinary text.
A block counts as open if the dedicated opening token appears anywhere, or if
the plain-text spelling is the very first thing in the draw, so the same words
later in a sentence stay prose. Closing markers count in either spelling
wherever they appear.

Normally a report must have one open block and exactly one close, and the
answer is everything after that close. A block opened and never closed, or
closed more than once, counts as unclosed and carries no answer. A draw with no
markers at all is read whole.

When the draw ended on its own rather than by running out of room, the reading
relaxes: several blocks are read from after the last close, and an unclosed
draw is read whole but still marked unclosed. A draw that hit the ceiling keeps
the strict rule. Either way the rule looks only at the markers and at how the
draw ended, never at what the draw says.
