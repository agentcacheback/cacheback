"""Selector callables: sender state, request IDs and position budget to selected indices."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, TypeAlias

import torch

from rcc.selectors.cacheback import cacheback
from rcc.selectors.chunkkv import chunkkv
from rcc.selectors.qsnap import qsnap

if TYPE_CHECKING:
    from rcc.transport import SenderState

Selector: TypeAlias = Callable[["SenderState", torch.Tensor, int], Sequence[int] | torch.Tensor]

__all__ = ["Selector", "cacheback", "chunkkv", "qsnap"]
