"""Family-owned score evidence behind the shared producer and audit route."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import torch

    from rcc.models.qwen.engine import WorkerProduct


class ScoreAdapter(Protocol):
    """Prepare, validate and replay one registered physical selector's evidence."""

    recipe: str
    score_schema: str
    peak_scope: str

    def validate_tokenizer(self, tokenizer: Any) -> None:
        """Require the tokenizer capabilities needed to prepare score evidence."""
        ...

    def shape(self, length: int) -> tuple[int, ...]:
        """Return the complete banked scoring tensor shape."""
        ...

    def prepare(
        self,
        product: WorkerProduct,
        tokenizer: Any,
        rendered: str,
        question: str,
        token_ids: Sequence[int],
    ) -> torch.Tensor:
        """Prepare replay evidence from one captured worker without consulting gold."""
        ...

    def validate(self, bank: torch.Tensor, base_score: torch.Tensor | None = None) -> None:
        """Validate the schema and optionally bind its original capture component."""
        ...

    def keep(self, bank: torch.Tensor, ratio: int) -> tuple[int, ...]:
        """Recompute an ordered keep from independent scoring evidence."""
        ...
