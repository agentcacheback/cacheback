"""Durable role-aware bank for Gemma text reports."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from rcc.run.runlog import DurableRunLog
from rcc.topologies.fanout import FANOUT_M3

# Every banked worker row carries the per-draw ladder (`draws_n`,
# `redraw_wall_s`) and the closer injection flag. A bank under another schema
# cannot rebuild the shipped span, so DurableRunLog refuses it by name.
TEXT_REPORT_BANK_SCHEMA = "gemma4-fanoutqa-text-report-bank-v2"


class TextReportBankProtocol(Protocol):
    """Replay and bank operations used by report generation."""

    def banked_report(self, qid: str, semantic_arm: str, worker: int) -> dict[str, Any] | None:
        """Return this worker's banked report, when one exists."""
        ...

    def bank_report(
        self,
        row: Mapping[str, Any],
        *,
        qid: str,
        semantic_arm: str,
        worker: int,
    ) -> dict[str, Any]:
        """Durably append one worker report with its role identity."""
        ...


class TextReportBank:
    """A role-aware durable report bank isolated from capture rows."""

    def __init__(
        self,
        path: Path,
        *,
        attempt_id: str,
        execution_identity: Mapping[str, Any],
        registration_fingerprint: str | None = None,
    ) -> None:
        """Open a bank, or reopen an existing one, under exact execution identities."""
        if registration_fingerprint is None:
            from rcc.models.gemma.text_contract import TEXT_REGISTRATION_FINGERPRINT

            registration_fingerprint = TEXT_REGISTRATION_FINGERPRINT
        if not attempt_id or len(registration_fingerprint) != 64:
            raise ValueError("Gemma text bank requires attempt and registration identities")
        frozen = {
            "text_report_bank_schema": TEXT_REPORT_BANK_SCHEMA,
            "text_registration_fingerprint": registration_fingerprint,
        }
        overlap = set(frozen) & set(execution_identity)
        if overlap:
            raise ValueError(f"execution identity overrides text bank fields: {sorted(overlap)!r}")
        self.attempt_id = attempt_id
        self.log = DurableRunLog(
            path,
            result_field="cell",
            fixed_fields={**frozen, **dict(execution_identity)},
        )

    @staticmethod
    def producer(semantic_arm: str, worker: int) -> str:
        """Return the durable producer key for one semantic worker."""
        if not semantic_arm or worker not in range(FANOUT_M3.workers_per_item):
            raise ValueError("Gemma report producer is outside the M=3 roster")
        return f"{semantic_arm}:w{worker}"

    def banked_report(self, qid: str, semantic_arm: str, worker: int) -> dict[str, Any] | None:
        """Return this worker's banked report, when one exists."""
        return self.log.reports.get((qid, self.producer(semantic_arm, worker)))

    def bank_report(
        self,
        row: Mapping[str, Any],
        *,
        qid: str,
        semantic_arm: str,
        worker: int,
    ) -> dict[str, Any]:
        """Durably append one worker report with its role identity."""
        return self.log.bank(
            {
                **dict(row),
                "kind": "producer",
                "qid": qid,
                "producer": self.producer(semantic_arm, worker),
                "semantic_arm": semantic_arm,
                "worker": worker,
            },
            attempt_id=self.attempt_id,
        )


__all__ = ("TEXT_REPORT_BANK_SCHEMA", "TextReportBank", "TextReportBankProtocol")
