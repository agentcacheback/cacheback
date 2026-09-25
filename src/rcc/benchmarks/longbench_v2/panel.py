"""Panel selection for the LongBench v2 chain-of-agents benchmark.

Selection reads the context token count, the domain, the difficulty, and the
question id, never the question, the choices, or the answer.
"""

from __future__ import annotations

import csv
import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rcc.benchmarks.protocol import BenchmarkProfile

DATASET_ID = "zai-org/LongBench-v2"
DATASET_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
DATASET_FILE = "data.json"
RAW_SOURCE_SHA256 = "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2"
RAW_SOURCE_BYTES = 465_490_535
RAW_ROWS = 503
TOKENIZER_CHECKPOINT = "Qwen/Qwen3-8B"
TOKENIZER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
TOKENIZER_ARTIFACT_SHA256 = {
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
UNMINTED_SENTINEL = "unminted-pending-longbench-coa-source-bundle"
CLUSTER_KEY = "sha256(utf8(context))"
EXECUTION_ORDER_RULE = "proportional-stratum-interleave-largest-remainder-smaller-band-first-v1"
SELECTION_BAND_TOKENS = 50_000
_PACKAGE = Path(__file__).parent


@dataclass(frozen=True)
class Panel:
    """One panel: its selection rule, its frozen files, and their digests."""

    benchmark_key: str
    sample_salt: str
    selection_rule: str
    #: Difficulties admitted before the one-per-context step; None admits every row.
    difficulties: tuple[str, ...] | None
    #: Sample size per 50K band; a band whose survivors fit its quota is taken whole.
    quotas: Mapping[int, int]
    #: Rows per stratum in the raw file before the one-per-context rule.
    expected_population: Mapping[int, int]
    #: Rows per stratum after the one-per-context rule.
    expected_survivors: Mapping[int, int]
    #: Rows the one-per-context rule removed from the strata.
    excluded_qids: tuple[str, ...]
    manifest_file: str
    manifest_sha256: str
    ledger_file: str
    ledger_sha256: str
    #: The passes declared over this panel's bank, as (start, count) pairs. A
    #: profile carries its own ``declared_passes``, and that is the set the
    #: resolver admits.
    registration_passes: tuple[tuple[int, int], ...]
    #: The longest question plus choices on the panel, in tokens, measured once.
    question_choices_max_tokens: int

    @property
    def size(self) -> int:
        """Return the number of items the quotas sum to."""
        return sum(self.quotas.values())

    @property
    def manifest_path(self) -> Path:
        """Return the frozen manifest beside this module."""
        return _PACKAGE / self.manifest_file

    @property
    def ledger_path(self) -> Path:
        """Return the frozen chunk ledger beside this module."""
        return _PACKAGE / self.ledger_file


#: The easy rows alone, over the same three strata. Quotas are the
#: largest-remainder split of the fifty over the easy survivors (29, 19, 10).
EASY50 = Panel(
    benchmark_key="longbench-v2-coa-easy50-sealed",
    sample_salt="longbench-v2-coa-easy50-v1",
    selection_rule="easy-only-one-question-per-context-lowest-selection-hash-v1",
    difficulties=("easy",),
    quotas={3: 25, 4: 16, 5: 9},
    expected_population={3: 30, 4: 21, 5: 10},
    expected_survivors={3: 29, 4: 19, 5: 10},
    excluded_qids=(
        "6724c83fbb02136c067d7962",
        "6725dda1bb02136c067d8654",
        "67286698bb02136c067d9223",
    ),
    manifest_file="easy50.csv",
    manifest_sha256="edee6376c10e73bfca396a376d2af7eee73205aba41bbdfff8da2d5de86db623",
    ledger_file="easy50_fixed4_chunks.jsonl",
    ledger_sha256="b88bc04c6615bda30fa2a018861b8628d13daab984dac66d1f200db02b1b7df1",
    registration_passes=((0, 50),),
    question_choices_max_tokens=883,
)
#: The re-ranked handoff law's config key over the easy fifty. It is a second key
#: onto the same panel object, not a second panel: the salt, the ledger, the
#: chunks, and the seed namespace are shared, so seeds pair item for item.
EASY50_RERANK_KEY = "longbench-v2-coa-easy50-rerank-sealed"
#: The text-only profile's config key over the easy fifty, a third key onto the
#: same panel object. The floor and the three text arms take the chunks, the
#: prompts, and the seed namespace the latent ladders use.
EASY50_TEXT_KEY = "longbench-v2-coa-easy50-text-sealed"
#: The bounded law's config key over the easy fifty, a fourth key onto the same
#: panel object.
EASY50_BOUNDED_KEY = "longbench-v2-coa-easy50-bounded-sealed"
#: The native Nemotron keys over the same easy fifty: the text baselines and
#: the latent ladders, each its own profile.
NEMOTRON_EASY50_TEXT_KEY = "longbench-v2-coa-nemotron-easy50-text-sealed"
NEMOTRON_EASY50_RERANK_KEY = "longbench-v2-coa-nemotron-easy50-rerank-sealed"
NEMOTRON_EASY50_BOUNDED_KEY = "longbench-v2-coa-nemotron-easy50-bounded-sealed"
NEMOTRON_PANEL_KEYS = frozenset(
    (NEMOTRON_EASY50_TEXT_KEY, NEMOTRON_EASY50_RERANK_KEY, NEMOTRON_EASY50_BOUNDED_KEY)
)
PANELS = {
    EASY50_RERANK_KEY: EASY50,
    EASY50_TEXT_KEY: EASY50,
    EASY50_BOUNDED_KEY: EASY50,
    NEMOTRON_EASY50_TEXT_KEY: EASY50,
    NEMOTRON_EASY50_RERANK_KEY: EASY50,
    NEMOTRON_EASY50_BOUNDED_KEY: EASY50,
}


def panel_for(profile: BenchmarkProfile) -> Panel:
    """Return the panel one profile names, raising when it names none."""
    try:
        return PANELS[profile.benchmark_key]
    except KeyError:
        raise ValueError(f"{profile.benchmark_key}: no sealed LongBench panel") from None


MANIFEST_FIELDS = (
    "sample_order",
    "qid",
    "selection_length_band_50k",
    "fixed_source_updates",
    "source_tokens",
    "domain",
    "sub_domain",
    "difficulty",
    "selection_hash",
    "worker_1_novel_tokens",
    "worker_2_novel_tokens",
    "worker_3_novel_tokens",
    "worker_4_novel_tokens",
    "max_worker_novel_tokens",
)
_UPDATES = 4


@dataclass(frozen=True)
class PanelRecord:
    """One raw row's selection-visible fields."""

    source_index: int
    qid: str
    domain: str
    sub_domain: str
    difficulty: str
    band: int
    source_tokens: int
    context_sha256: str
    selection_hash: str


@dataclass(frozen=True)
class PanelSelection:
    """The selected rows in panel order, with the counts the rule produced."""

    selected: tuple[PanelRecord, ...]
    excluded: tuple[str, ...]
    population: dict[int, int]
    survivors: dict[int, int]


def selection_band(source_tokens: int) -> int:
    """Return the 50K source-length band one context falls in."""
    if source_tokens < 1:
        raise ValueError("source tokens must be positive")
    return math.ceil(source_tokens / SELECTION_BAND_TOKENS)


def context_sha256(context: str) -> str:
    """Return the cluster key of one context."""
    return hashlib.sha256(context.encode("utf-8")).hexdigest()


def sample_hash(
    dataset_sha: str, *, salt: str, band: int, domain: str, difficulty: str, qid: str
) -> str:
    """Return the salted selection hash of one row."""
    material = "|".join((salt, dataset_sha, str(band), domain, difficulty, qid))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def hamilton_quotas(
    cells: Sequence[tuple[str, str]], sample_size: int
) -> dict[tuple[str, str], int]:
    """Allocate a sample over (domain, difficulty) cells by largest remainder."""
    counts = Counter(cells)
    total = len(cells)
    if sample_size > total:
        raise ValueError("a sample cannot exceed its population")
    raw = {cell: sample_size * count / total for cell, count in counts.items()}
    quotas = {cell: math.floor(value) for cell, value in raw.items()}
    remaining = sample_size - sum(quotas.values())
    order = sorted(raw, key=lambda cell: (-(raw[cell] - quotas[cell]), cell))
    for cell in order[:remaining]:
        quotas[cell] += 1
    if sum(quotas.values()) != sample_size:
        raise RuntimeError("Hamilton allocation did not close")
    if any(quotas[cell] > counts[cell] for cell in quotas):
        raise RuntimeError("a sample quota exceeds its source cell")
    return quotas


def panel_records(
    rows: Sequence[Mapping[str, Any]],
    source_tokens: Mapping[str, int],
    *,
    salt: str,
    dataset_sha: str = RAW_SOURCE_SHA256,
) -> tuple[PanelRecord, ...]:
    """Reduce raw rows to their selection-visible fields, one record per row."""
    records: list[PanelRecord] = []
    for index, row in enumerate(rows):
        qid = str(row["_id"])
        tokens = int(source_tokens[qid])
        band = selection_band(tokens)
        domain, difficulty = str(row["domain"]), str(row["difficulty"])
        records.append(
            PanelRecord(
                source_index=index,
                qid=qid,
                domain=domain,
                sub_domain=str(row["sub_domain"]),
                difficulty=difficulty,
                band=band,
                source_tokens=tokens,
                context_sha256=context_sha256(str(row["context"])),
                selection_hash=sample_hash(
                    dataset_sha,
                    salt=salt,
                    band=band,
                    domain=domain,
                    difficulty=difficulty,
                    qid=qid,
                ),
            )
        )
    if len({record.qid for record in records}) != len(records):
        raise ValueError("raw rows repeat a question id")
    return tuple(records)


def select_panel(records: Sequence[PanelRecord], panel: Panel) -> PanelSelection:
    """Admit the panel's difficulties, apply one-per-context, then the quotas per stratum."""
    quotas = panel.quotas
    if panel.difficulties is not None:
        records = [record for record in records if record.difficulty in panel.difficulties]
    population = Counter(record.band for record in records if record.band in quotas)
    best: dict[str, PanelRecord] = {}
    for record in records:
        current = best.get(record.context_sha256)
        if current is None or record.selection_hash < current.selection_hash:
            best[record.context_sha256] = record
    winners = set(best.values())
    excluded = tuple(
        sorted(record.qid for record in records if record not in winners and record.band in quotas)
    )
    survivors = [record for record in winners if record.band in quotas]
    selected: list[PanelRecord] = []
    for band, size in quotas.items():
        candidates = [record for record in survivors if record.band == band]
        if len(candidates) <= size:
            selected.extend(candidates)
            continue
        cells = hamilton_quotas([(r.domain, r.difficulty) for r in candidates], size)
        by_cell: dict[tuple[str, str], list[PanelRecord]] = defaultdict(list)
        for record in candidates:
            by_cell[(record.domain, record.difficulty)].append(record)
        for cell, quota in sorted(cells.items()):
            selected.extend(sorted(by_cell[cell], key=lambda r: r.selection_hash)[:quota])
    selected.sort(key=lambda record: (record.band, record.selection_hash, record.qid))
    return PanelSelection(
        selected=tuple(selected),
        excluded=excluded,
        population={band: population[band] for band in quotas},
        survivors={band: sum(record.band == band for record in survivors) for band in quotas},
    )


def validate_selection(selection: PanelSelection, panel: Panel) -> None:
    """Raise when a selection differs from the panel it was drawn for."""
    if selection.population != panel.expected_population:
        raise RuntimeError(f"unexpected stratum population: {selection.population}")
    if selection.survivors != panel.expected_survivors:
        raise RuntimeError(f"unexpected one-per-context survivors: {selection.survivors}")
    if selection.excluded != panel.excluded_qids:
        raise RuntimeError("the one-per-context rule excluded a different set of rows")
    selected, size = selection.selected, panel.size
    if len(selected) != size or len({record.qid for record in selected}) != size:
        raise RuntimeError(f"the panel must contain {size} unique question ids")
    if len({record.context_sha256 for record in selected}) != size:
        raise RuntimeError(f"the panel must contain {size} distinct contexts")
    if Counter(record.band for record in selected) != Counter(panel.quotas):
        raise RuntimeError("the panel strata differ from the sealed quotas")


def execution_order(
    bands_by_qid: Mapping[str, int], ordered_qids: Sequence[str]
) -> tuple[str, ...]:
    """Interleave the strata proportionally, largest remainder per prefix.

    Ties go to the smaller band, so every prefix of the order holds each
    stratum as close to its share as whole items allow.
    """
    strata: dict[int, list[str]] = defaultdict(list)
    for qid in ordered_qids:
        strata[bands_by_qid[qid]].append(qid)
    quota = {band: len(qids) for band, qids in strata.items()}
    total = sum(quota.values())
    taken = dict.fromkeys(strata, 0)
    order: list[str] = []
    for index in range(1, total + 1):
        band = max(strata, key=lambda band: (quota[band] * index / total - taken[band], -band))
        order.append(strata[band][taken[band]])
        taken[band] += 1
    return tuple(order)


def read_manifest(path: Path) -> tuple[dict[str, str], ...]:
    """Read the frozen manifest rows as strings, in sample order, checking the columns."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = tuple(reader)
    if reader.fieldnames != list(MANIFEST_FIELDS) or not rows:
        raise RuntimeError(f"{path}: manifest columns differ from the sealed layout")
    return rows


def manifest_rows(
    selection: PanelSelection, chunk_tokens: Mapping[str, Sequence[int]]
) -> list[dict[str, object]]:
    """Render the manifest rows for one selection and its chunk token counts."""
    rows: list[dict[str, object]] = []
    for order, record in enumerate(selection.selected, start=1):
        loads = [int(count) for count in chunk_tokens[record.qid]]
        if len(loads) != _UPDATES:
            raise ValueError(f"{record.qid}: expected {_UPDATES} chunk token counts")
        row: dict[str, object] = {
            "sample_order": order,
            "qid": record.qid,
            "selection_length_band_50k": record.band,
            "fixed_source_updates": _UPDATES,
            "source_tokens": record.source_tokens,
            "domain": record.domain,
            "sub_domain": record.sub_domain,
            "difficulty": record.difficulty,
            "selection_hash": record.selection_hash,
        }
        for worker, load in enumerate(loads, start=1):
            row[f"worker_{worker}_novel_tokens"] = load
        row["max_worker_novel_tokens"] = max(loads)
        rows.append(row)
    return rows


def write_manifest(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write manifest rows in the frozen column order and byte layout.

    The frozen bytes end every line with CR LF, the csv module's default, so the
    terminator is not overridden here.
    """
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in MANIFEST_FIELDS})


def strata_counts(rows: Sequence[Mapping[str, object]]) -> dict[int, int]:
    """Count manifest rows per 50K band."""
    counts = Counter(int(str(row["selection_length_band_50k"])) for row in rows)
    return {band: counts[band] for band in sorted(counts)}


__all__ = (
    "CLUSTER_KEY",
    "DATASET_FILE",
    "DATASET_ID",
    "DATASET_REVISION",
    "EASY50",
    "EASY50_BOUNDED_KEY",
    "EASY50_RERANK_KEY",
    "EASY50_TEXT_KEY",
    "EXECUTION_ORDER_RULE",
    "MANIFEST_FIELDS",
    "NEMOTRON_EASY50_BOUNDED_KEY",
    "NEMOTRON_EASY50_RERANK_KEY",
    "NEMOTRON_EASY50_TEXT_KEY",
    "NEMOTRON_PANEL_KEYS",
    "PANELS",
    "RAW_ROWS",
    "RAW_SOURCE_BYTES",
    "RAW_SOURCE_SHA256",
    "SELECTION_BAND_TOKENS",
    "TOKENIZER_ARTIFACT_SHA256",
    "TOKENIZER_CHECKPOINT",
    "TOKENIZER_REVISION",
    "UNMINTED_SENTINEL",
    "Panel",
    "PanelRecord",
    "PanelSelection",
    "context_sha256",
    "execution_order",
    "hamilton_quotas",
    "manifest_rows",
    "panel_for",
    "panel_records",
    "read_manifest",
    "sample_hash",
    "select_panel",
    "selection_band",
    "strata_counts",
    "validate_selection",
    "write_manifest",
)
