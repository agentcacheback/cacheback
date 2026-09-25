"""Owner-run GPU release gate for a real vLLM sender-to-receiver workflow."""

import os

import pytest


@pytest.mark.gpu
def test_vllm_capture_transfer_and_receiver_continuation() -> None:
    from examples.vllm_transfer import run

    outputs = run(allow_unstable=os.environ.get("RCC_TEST_UNSTABLE") == "1")
    assert len(outputs) == 68 and all(len(tokens) == 4 for tokens in outputs)
