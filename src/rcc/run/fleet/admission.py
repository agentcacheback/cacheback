"""The KV token reservation every fleet receiver admits requests under.

A request is admitted only when its prompt plus the whole answer ceiling still
fits, and its reservation is released when it finishes.
"""

from __future__ import annotations


class AdmissionGate:
    """Reserve the prompt plus the whole answer ceiling for each live request."""

    def __init__(
        self,
        pool_tokens: int,
        *,
        answer_ceiling: int,
        safety: float = 0.95,
        max_in_flight: int = 16,
    ) -> None:
        """Configure the token pool and per-request output ceiling."""
        if pool_tokens <= 0 or answer_ceiling <= 0:
            raise ValueError("pool_tokens and answer_ceiling must be positive")
        if not 0.0 < safety <= 1.0:
            raise ValueError(f"safety must be in (0, 1], got {safety}")
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be positive")
        self.capacity_tokens = int(pool_tokens * safety)
        self.answer_ceiling = answer_ceiling
        self.max_in_flight = max_in_flight
        self._reserved: dict[str, int] = {}

    @property
    def reserved_tokens(self) -> int:
        """Return the KV tokens currently reserved by live requests."""
        return sum(self._reserved.values())

    def request_bound(self, prompt_tokens: int) -> int:
        """Return one request's prompt-plus-output KV reservation."""
        if prompt_tokens <= 0:
            raise ValueError(f"prompt_tokens must be positive, got {prompt_tokens}")
        bound = prompt_tokens + self.answer_ceiling
        if bound > self.capacity_tokens:
            raise ValueError(
                f"a single request needs {bound} tokens against capacity "
                f"{self.capacity_tokens}; the spec would deadlock the gate"
            )
        return bound

    def try_admit(self, request_id: str, prompt_tokens: int) -> bool:
        """Reserve one request, or return False while it must remain queued."""
        if request_id in self._reserved:
            raise ValueError(f"request {request_id!r} is already admitted")
        bound = self.request_bound(prompt_tokens)
        if len(self._reserved) >= self.max_in_flight:
            return False
        if self.reserved_tokens + bound > self.capacity_tokens:
            return False
        self._reserved[request_id] = bound
        return True

    def extend(self, request_id: str, tokens: int) -> None:
        """Book extra output tokens onto a live request's reservation.

        The ceiling bounds the head, not the closer's continuation, so a tail can
        push the reserved total over capacity by at most one tail per live head.
        """
        if request_id not in self._reserved:
            raise ValueError(f"request {request_id!r} was never admitted")
        if tokens <= 0:
            raise ValueError(f"extension must be positive, got {tokens}")
        self._reserved[request_id] += int(tokens)

    def release(self, request_id: str) -> None:
        """Release one finished request's reservation."""
        if request_id not in self._reserved:
            raise ValueError(f"request {request_id!r} was never admitted")
        del self._reserved[request_id]
