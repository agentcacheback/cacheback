"""One mathematical oracle for support correction and the protected span boundary."""

import torch

from rclc.selectors.cacheback import support_score
from rclc.selectors.fixed_spans import fixed_span_keep


def test_support_dose_and_zero_energy_have_known_answers() -> None:
    votes = torch.tensor([0.25, 0.4, 0.2, *[0.0] * 4])
    energy = torch.tensor([0.0, 0.16, 0.04, *[1.0] * 4])
    row_energy = torch.tensor([0.0, 0.32, 0.04, *[1.0] * 4])
    torch.testing.assert_close(
        support_score(votes, energy, row_energy)[:3], torch.tensor([0.4, 1.6, 0.4])
    )
    scores = torch.ones(66)
    scores[16:32] = 10
    assert fixed_span_keep(scores, 19, (0, 64, 65), span_size=16) == [0, *range(16, 32), 64, 65]
    assert fixed_span_keep(scores, 20, (0, 64, 65), span_size=16) == [0, 1, *range(16, 32), 64, 65]
