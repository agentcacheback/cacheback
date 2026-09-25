"""The Ministral binding of the shared split-bank report.

The report itself is shared. Ministral contributes the schema its rows are bound
to and the identity fields that say which run a reader is holding.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from rcc.models.ministral import MINISTRAL, MINISTRAL_RUNTIME
from rcc.models.ministral.payload import MINISTRAL_PAYLOAD_LAYOUT
from rcc.models.ministral.results import MINISTRAL_RESULT_SCHEMA
from rcc.run.barrier import SPLIT_FLEET_ISOLATION_PROFILE
from rcc.run.fleet.merge import CompletionKey
from rcc.run.fleet.report import SplitReportAdapter, SplitReportSpec

MINISTRAL_SPLIT_REPORT_SCHEMA = "ministral-fanoutqa-split-report-v1"


def ministral_split_report_spec(*, unified_plan_fingerprint: str) -> SplitReportSpec:
    """Return the identity a Ministral split report is published under."""
    return SplitReportSpec(
        schema=MINISTRAL_SPLIT_REPORT_SCHEMA,
        label="Ministral",
        identity={
            "row_schema": MINISTRAL_RESULT_SCHEMA,
            "payload_layout": MINISTRAL_PAYLOAD_LAYOUT,
            "isolation_profile": SPLIT_FLEET_ISOLATION_PROFILE,
            "unified_plan_fingerprint": unified_plan_fingerprint,
            "runtime_profile_fingerprint": MINISTRAL_RUNTIME.runtime_identity_hash,
            "decode_profile": MINISTRAL.decode.profile_id,
            "decode_fingerprint": MINISTRAL.decode.identity_hash,
        },
    )


def ministral_split_report_adapter(
    placements: Any,
    completion_key: CompletionKey,
    *,
    unified_plan_fingerprint: str,
) -> SplitReportAdapter:
    """Return the shared split report bound to the Ministral run identity.

    The spec factory is a partial rather than a closure because the adapter holding
    it crosses a process boundary and reaches a spawned child by pickle.
    """
    return SplitReportAdapter(
        partial(
            ministral_split_report_spec,
            unified_plan_fingerprint=unified_plan_fingerprint,
        ),
        placements,
        completion_key,
    )


__all__ = (
    "MINISTRAL_SPLIT_REPORT_SCHEMA",
    "ministral_split_report_adapter",
    "ministral_split_report_spec",
)
