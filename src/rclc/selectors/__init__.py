"""Selector callables: sender state, request IDs and position budget to selected indices."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, TypeAlias

import torch

from rclc.selectors.cacheback import cacheback
from rclc.selectors.chunkkv import chunkkv
from rclc.selectors.qsnap import qsnap

if TYPE_CHECKING:
    from rclc.transport import SenderState

Selector: TypeAlias = Callable[["SenderState", torch.Tensor, int], Sequence[int] | torch.Tensor]

__all__ = ["Selector", "cacheback", "chunkkv", "qsnap"]
