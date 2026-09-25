# Replicate the paper

The `paper` branch preserves the benchmark implementation. `paper-v1` fixes
the first replication snapshot; `main` provides the state-transfer API.
Research-only selectors, KV transforms, capture-bank replay and their original
tests remain on `paper`. They are outside the public transport API on `main`.

In a separate checkout, select the snapshot:

```bash
git switch --detach paper-v1
```

Follow that snapshot's README for installation and the exact benchmark commands.
It includes:

- Twelve configs for FanOutQA and LongBench v2, including the request ablation.
- Qwen 3, Gemma 4, Ministral 3, and Nemotron serving runtimes.
- A requirements file for each of the two measured vLLM stacks.
- The natural FanOutQA and selector-development archives under `data/`.
- CPU tests, result scoring, and handoff-recall analysis.

The bundled archives unpack offline with digest verification. Model weights
and the pinned LongBench dataset download separately. Gemma and Ministral
require accepted model access terms. The registered benchmark runs use one
node with eight H100 80 GB GPUs; the API's small-model example does
not reproduce their accuracy or latency.

The manuscript is *Receiver-Conditioned Latent Communication gives 94%
CacheBack*. **Paper link: TBA.**
