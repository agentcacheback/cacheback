# Contributing

## Bring your own selector

The challenge: improve answer quality at the same communication budget, or
reduce selection cost while retaining quality. CacheBack is the default;
selection is a callable so you can try another method.

### 1. Add your selector

Follow the [selector contract and example](docs/selectors.md). Your function
receives sender state, request IDs and a resolved position budget, and returns
selected indices:

```python
rcc.transfer(senders, receivers, requests, selector=my_selector)
```

For a contribution to the package, put the function in `src/rcc/selectors/`.
Keep it small and leave sender state unchanged. Extend an existing end-to-end
scenario to cover selection, delivery and receiver continuation; avoid a test
per helper. Follow the [development harness](docs/development.md) and run:

```bash
.venv/bin/python scripts/check.py
```

### 2. Compare on the paper's benchmarks

Use a separate checkout and environment for the frozen `paper-v1` snapshot.
The [replication guide](docs/replication.md) covers setup. In that snapshot,
`docs/benchmarks.md` describes the panels and `docs/running.md` explains runs
and reports.

| Benchmark | Task | Qwen baseline config on `paper` |
| --- | --- | --- |
| FanOutQA | Three senders deliver evidence to one receiver | `configs/qwen-fanoutqa-natural-dev50.toml` |
| LongBench v2 | Pass and rerank state through a four-chunk chain | `configs/qwen-longbench-coa-easy50-rerank-n50.toml` |
| LongBench v2, bounded | Same chain with a fixed position cap | `configs/qwen-longbench-coa-easy50-bounded-n50.toml` |

For example, after installing the paper's Qwen environment, reproduce its
FanOutQA baseline with:

```bash
python -m rcc.data fetch fanoutqa-natural-dev50
python -m rcc.run.entry --config configs/qwen-fanoutqa-natural-dev50.toml \
    --bundle data/fanoutqa-natural-dev50
```

These registered runs use eight H100 80 GB GPUs. The CPU quickstart checks API
usage; it does not reproduce benchmark results.

**Custom selector integration into the paper benchmarks is currently manual.**
The frozen runner does not accept the main branch's `selector=` callable directly.
Start an experiment branch from `paper-v1`, integrate your method into the
selected model's selection path, and register a distinct experimental arm and
run. Preserve the frozen branch, tag and baseline configs.

Keep the evaluation panel, model, prompts, generation settings, seeds and
scoring fixed between methods. Match relative `rX` or bounded budgets, including
inherited positions in chains, and report actual retained positions and payload
bytes. Keep tuning separate from the evaluation panels. The paper snapshot's
`docs/selector-dev.md` describes the development panel; replaying its stored
keep sets only reproduces existing selectors, not a new method.

### 3. Share the comparison

Include the selector code and experiment integration, exact revision and run
commands, configs, hardware/runtime pins and raw reports. Compare answer accuracy,
retained positions, payload bytes and selection time against CacheBack at the
same budgets. If reporting completion latency, match hardware and concurrency
and include selection and handoff costs. Label small runs and untested settings
clearly. A useful contribution does not need to beat CacheBack.

## Bring a new benchmark

New tasks, datasets and agent communication patterns are welcome, including
cases where CacheBack struggles. You can contribute a benchmark without adding
a selector. Use the public API in a runnable experiment, or extend the research
runner on a separate experiment branch; keep the `paper` snapshot frozen.

Include a reproducible data source and its license, fixed development/evaluation
splits, sender/receiver setup, requests and scoring code. Provide one small
end-to-end example and commands for the full evaluation. Compare CacheBack with
a relevant baseline under matched conditions, reporting task quality,
communication budgets and costs using the reporting guidance above. Explain
what the benchmark tests that the existing panels do not.
