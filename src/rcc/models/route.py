"""One capture route, several model families: the facts the Qwen route reads.

The route is one body of code under ``rcc.models.qwen`` and ``rcc.run.qwen``.
Here is the family contract it takes; it reads no family constant of its own.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rcc.benchmarks.protocol import BenchmarkProfile
from rcc.models.protocol import ModelProfile
from rcc.models.selection import ScoreAdapter

#: Decodes banked ids to text with special tokens kept and no cleanup, the
#: tokenizer's own ``decode`` bound by the caller through ``tokenizer_decoder``.
Decoder = Callable[[Sequence[int]], str]


def _no_overrides() -> dict[str, Any]:
    return {}


TEXT_SEMANTIC_ARMS = ("text_primary", "text_medium", "text_small")


def tokenizer_decoder(tokenizer: Any) -> Decoder:
    """Bind one tokenizer's raw decode, the reading every block predicate takes.

    Special tokens are kept: a family whose delimiters are reserved ids renders
    them only under this flag, and prose delimiters are unaffected by it.
    """

    def decode(token_ids: Sequence[int]) -> str:
        return str(
            tokenizer.decode(
                [int(token) for token in token_ids],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )

    return decode


@dataclass(frozen=True)
class RouteFamily:
    """One model family's facts for the shared Qwen route."""

    profile: ModelProfile
    #: Lane word for run ids, storage roots, ``--family``, and schema names.
    lane: str
    #: Prefix every physical policy of this family carries, ``none`` aside.
    policy_prefix: str
    #: The transformers class the engine's fused weights alias into.
    hf_architecture: str
    stop_token_ids: tuple[int, ...]
    think_open: str
    think_close: str
    #: Ids that spell the opener and the closer: one reserved id each for a
    #: family whose tokenizer reserves them, the literal spelling's ids for a
    #: family that writes the tags as ordinary text.
    think_open_token_ids: tuple[int, ...]
    think_close_token_ids: tuple[int, ...]
    #: True when the tags are ordinary text. Block state is then read from the
    #: decoded draw, because prose ids cannot be told apart from tag ids.
    literal_think_tags: bool
    #: The system message that switches thinking on for a family whose chat
    #: template has no thinking flag. None means the template's own kwarg.
    thinking_system_prompt: str | None = None
    #: vLLM ``hf_overrides`` for the engine (the Qwen YaRN window). Empty when
    #: the checkpoint config already states the served window.
    hf_overrides: Mapping[str, Any] = field(default_factory=_no_overrides)
    #: How far a draw may decode after its closer is injected. It is not part
    #: of the decode or runtime fingerprint.
    closing_token_budget: int = 4096
    #: Optional template controls retained in native sender artifact identities.
    assistant_prefill: str | None = None
    low_effort: bool = False
    #: Place latent payload rows inside the user turn, at the slot where the
    #: text arms render their worker reports, instead of before the system
    #: header. Off by default; the Nemotron family declares it.
    payload_in_user_turn: bool = False

    def __post_init__(self) -> None:
        """Refuse a family whose facts cannot drive the route."""
        if not self.lane or not self.lane.isalnum() or not self.lane.islower():
            raise ValueError("route lane must be one lowercase alphanumeric word")
        if not self.stop_token_ids:
            raise ValueError(f"{self.lane}: route family needs at least one stop id")
        if not self.think_open_token_ids or not self.think_close_token_ids:
            raise ValueError(f"{self.lane}: route family needs opener and closer ids")
        if not self.literal_think_tags and (
            len(self.think_open_token_ids) != 1 or len(self.think_close_token_ids) != 1
        ):
            raise ValueError(f"{self.lane}: reserved thinking delimiters are one id each")
        if not self.profile.decode.enable_thinking:
            raise ValueError(f"{self.lane}: the route decodes with thinking enabled")
        if self.closing_token_budget < 1:
            raise ValueError(f"{self.lane}: closing token budget must be positive")
        for arm in self.profile.physical_arms:
            if arm.policy != "none" and not arm.policy.startswith(self.policy_prefix):
                raise ValueError(f"{self.lane}: policy {arm.policy!r} lacks the family prefix")
        missing = [arm for arm in TEXT_SEMANTIC_ARMS if arm not in self._policy_by_arm()]
        if missing:
            raise ValueError(f"{self.lane}: route family lacks text arms {missing}")
        self._validate_execution_order()

    def _validate_execution_order(self) -> None:
        if self.profile.execution_order and (
            len(self.profile.execution_order) != len(self.profile.physical_arms)
            or set(self.profile.execution_order) != set(self._policy_by_arm())
        ):
            raise ValueError("production execution order must cover every arm exactly once")

    @property
    def production_policies(self) -> tuple[str, ...]:
        """Return the registered execution priority independently of semantic table order."""
        order = self.profile.execution_order or tuple(self._policy_by_arm())
        return tuple(self.policy(arm) for arm in order)

    @property
    def model_id(self) -> str:
        """Return the registered model id."""
        return self.profile.model_id

    @property
    def native_sender_prompts(self) -> bool:
        """Return whether each sender requires its own sealed prompt artifact."""
        return False

    def sender_family(self, semantic_arm: str) -> RouteFamily:
        """Return the route family one sender arm decodes under."""
        return self

    def benchmark_profile(self, profile: BenchmarkProfile) -> BenchmarkProfile:
        """Resolve family-specific output budgets, returning the profile unchanged."""
        return profile

    def score_adapter(self, semantic_arm: str) -> ScoreAdapter | None:
        """Resolve optional family-owned scoring evidence; default is the shared vector."""
        arm = next(a for a in self.profile.physical_arms if a.semantic_arm == semantic_arm)
        if arm.selector_recipe is not None:
            raise ValueError("registered selector recipe lacks its family implementation")
        return None

    @property
    def run_prefix(self) -> str:
        """Return the registered run-id prefix for this lane."""
        return f"{self.lane}-fanoutqa-m3-n15-v1-"

    def _policy_by_arm(self) -> dict[str, str]:
        return {arm.semantic_arm: arm.policy for arm in self.profile.physical_arms}

    def policy(self, semantic_arm: str) -> str:
        """Return the physical policy that implements one semantic arm."""
        try:
            return self._policy_by_arm()[semantic_arm]
        except KeyError:
            raise ValueError(f"{self.lane}: unregistered semantic arm {semantic_arm!r}") from None

    def semantic_arm(self, policy: str) -> str:
        """Return the semantic arm one physical policy implements."""
        for arm in self.profile.physical_arms:
            if arm.policy == policy:
                return arm.semantic_arm
        raise ValueError(f"{self.lane}: unregistered policy {policy!r}")

    @property
    def text_policies(self) -> tuple[str, ...]:
        """Return the primary, medium, and small sender policies in that order."""
        return tuple(self.policy(arm) for arm in TEXT_SEMANTIC_ARMS)

    @property
    def prefilled_block(self) -> bool:
        """Return whether the assistant prefill already opened the thinking block."""
        return self.assistant_prefill is not None and self.think_open in self.assistant_prefill

    def block_opened(self, token_ids: Sequence[int], *, decode: Decoder) -> bool:
        """Return whether a draw sits inside a thinking block it or the prefill opened."""
        if self.prefilled_block:
            return True
        if self.literal_think_tags:
            return self.think_open in decode(token_ids)
        return self.think_open_token_ids[0] in frozenset(int(token) for token in token_ids)

    def block_closed(self, token_ids: Sequence[int], *, decode: Decoder) -> bool:
        """Return whether a draw wrote a thinking closer."""
        if self.literal_think_tags:
            return self.think_close in decode(token_ids)
        return self.think_close_token_ids[0] in frozenset(int(token) for token in token_ids)

    def thinking_is_unclosed(self, token_ids: Sequence[int], *, decode: Decoder) -> bool:
        """Return whether a draw ended inside a block it opened but never closed.

        This triggers closer injection on the report and answer sides. Reserved
        delimiters are read as ids, literal tags from the decoded draw.
        """
        return self.block_opened(token_ids, decode=decode) and not self.block_closed(
            token_ids, decode=decode
        )

    def post_think_report(self, raw_output: str) -> str:
        """Return a draw's visible text, the part that follows its thinking.

        A model that opens its own block is read after the first closer, a
        prompt-opened block after the last. An unclosed draw has no visible text.
        """
        if self.think_close not in raw_output:
            return ""
        if not self.prefilled_block:
            return raw_output.split(self.think_close, 1)[1].strip()
        return raw_output.rpartition(self.think_close)[2].strip()

    def post_think_handoff(self, raw_output: str, *, ended: bool) -> str:
        """Return the visible report after a closed thinking block."""
        return self.post_think_report(raw_output)

    def report_is_accepted(self, raw_output: str, *, ended: bool = True) -> bool:
        """Apply the route's rule: any nonempty post-think report passes."""
        return bool(self.post_think_handoff(raw_output, ended=ended))


__all__ = ("TEXT_SEMANTIC_ARMS", "Decoder", "RouteFamily", "tokenizer_decoder")
