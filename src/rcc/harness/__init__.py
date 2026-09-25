"""The model-in-the-loop layer around the pure cache transforms.

`handoff` is re-exported here; the scoring utilities stay under
`rcc.harness.score`.
"""

from rcc.harness import score as score
from rcc.harness.handoff import handoff

__all__ = ["handoff", "score"]
