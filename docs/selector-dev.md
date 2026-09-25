# The selector-development experiment

Paper Section 3.2 and Appendix D compare CacheBack against mean query attention
and four published baselines on evidence recall, and report how well mean
query attention scores align with the gold evidence. Both come from one
capture-only sweep:
thirty frozen FanOutQA development questions, prefilled unpadded at about
fifty thousand tokens each, rolled forty latent steps, scored once by every
selector of the roster, and cut at ratios 2 through 128. Nothing is decoded.

The paper reports the thirteen questions whose required evidence is fully located
on the bound page. Eligibility is a property of the input, not of a selector or
a keep set: a question counts when every required gold leaf has an occurrence
the importer could locate in the prompt. Both scripts recompute it from
`gold_leaf_token_spans` in `run/inputs/<qid>.json` before they open anything
else, so the seventeen ineligible questions are never scored. Their inputs ship
anyway, because the plan's identity covers all thirty.

## Regenerating the tables on CPU

```bash
python -m rcc.data fetch selector-dev-v8
python -m experiments.selector_dev_recall data/selector-dev-v8/run
python -m experiments.selector_dev_alignment data/selector-dev-v8/run
```

The first unpacks `data/selector-dev-v8.tar.zst` against its pinned digest and
needs no network. The second writes `recall.csv`, `recall-table.csv`,
`recall-table.tex` and the parameter-sensitivity grid under `output/selector-dev/`.
`support-grid-recall.csv` and `support-grid-recall.tex` contain the values for
the paper's Table 4. The third writes `attention.csv`, whose `median_percentile`
column supplies Figure 4d.

The bundle carries every selector's keep sets, so the grid tables regenerate in
full. Only the mean query attention score vector ships, under the key `snap`
in each `scores.pt`: it is the one the alignment panel reads, and the other
thirty-three would have made the archive two orders of magnitude larger. The
digest `capture.json` records for `scores.pt` is therefore the digest of the
unreduced capture file, not of the file beside it.

## Rerunning the capture on GPU

Use the Qwen 3 environment described in the README. Before importing inputs,
download the tokenizer files for all four pinned revisions in
`src/rcc/run/selector_dev/contract.py`. Before capturing, also download the
weights and generation configuration for each model you intend to run. The
import and capture commands use the local Hugging Face cache and run offline.
For example, to prepare the tokenizers and the 1.7B capture model while online:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
from rcc.run.selector_dev.contract import MODELS

for repository, revision in MODELS.values():
    snapshot_download(repository, revision=revision,
                      allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken"])
repository, revision = MODELS["qwen3-1.7b"]
snapshot_download(repository, revision=revision)
PY
```

Then import and capture. Replace `SOURCE_COMMIT` with the full 40-character
source commit; when running from a Git checkout, omit `--source-commit` to
record the checkout's commit automatically.

```bash
python -m rcc.data fetch selector-dev-v8
python -m rcc.run.selector_dev import --root runs/selector-dev \
    --bundle data/selector-dev-v8 --source-commit SOURCE_COMMIT
python -m rcc.run.selector_dev check --root runs/selector-dev
CUDA_VISIBLE_DEVICES=0 python -m rcc.run.selector_dev capture --root runs/selector-dev \
    --model qwen3-1.7b --seat 0
```

`import` reads the audited prompts from `<bundle>/prepared`, checks their bytes
against their audit manifest, re-encodes every prompt with all four pinned
Qwen3 tokenizers, and writes an immutable plan into a fresh `--root`. It needs
the four tokenizer snapshots in the local Hugging Face cache, since it opens
them with `local_files_only`. It normally pins the commit of the checkout it
runs from; `--source-commit` supplies that commit when there is no checkout, as
in an unpacked supplement.

Run `capture` once per size, with `--seat` distinguishing concurrent seats of
one size. Select the physical GPU with `CUDA_VISIBLE_DEVICES`; `--seat` only
labels the process. Each process must own exactly one H100 or H200: the 1.7B,
4B and 8B sizes fit an 80 GB H100, and `engine.py` refuses `qwen3-32b` below 128 GB of
device memory, so the largest size needs an H200. A capture writes
`scores.pt`, `keeps.json`, `embeddings.pt` and a `capture.json` receipt per
question. The two analysis scripts read that run root directly; they do not
require decoding or the embedding files.

## What does not rebuild here

The thirty prompts do not rebuild from source in this tree. They were cut from
dated Wikipedia revisions by the private preparation pipeline, audited there,
and are shipped as prepared token ids with their audit manifest. This
repository can check them, import them and capture over them; it cannot rebuild
them again from the pages.
