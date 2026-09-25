"""One mathematical oracle for support correction and the protected span boundary."""

import torch

from rcc.selectors.fixed_spans import fixed_span_keep
from rcc.selectors.query_support.methods.compose import SupportMomentBundle, compose_score


def test_support_dose_and_zero_energy_have_known_answers() -> None:
    bundle = SupportMomentBundle(
        snap=torch.tensor([0.25, 0.4, 0.2]),
        e={2.0: torch.tensor([0.0, 0.16, 0.04])},
        eprime={2.0: torch.tensor([0.0, 0.16, 0.04])},
        r={2.0: torch.tensor([0.0, 0.32, 0.04])},
        einf=torch.zeros(3),
        rinf=torch.zeros(3),
    )
    torch.testing.assert_close(
        compose_score(bundle, order=2.0, alpha=2), torch.tensor([0.25, 1.6, 0.2])
    )
    scores = torch.ones(66)
    scores[16:32] = 10
    assert fixed_span_keep(scores, 19, (0, 64, 65), span_size=16) == [0, *range(16, 32), 64, 65]
    assert fixed_span_keep(scores, 20, (0, 64, 65), span_size=16) == [0, 1, *range(16, 32), 64, 65]
