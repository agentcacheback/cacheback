"""Await blocking model work without abandoning its cleanup on cancellation."""

import asyncio
import inspect
from collections.abc import Callable
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")


async def run_in_worker(function: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs) -> _T:
    """Wait for a started worker to restore state before propagating cancellation."""
    work = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(work)
    except asyncio.CancelledError:
        while not work.done():
            try:
                await asyncio.shield(work)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not work.cancelled() and work.exception() is None:
            result = work.result()
            if inspect.iscoroutine(result):
                result.close()
        raise
