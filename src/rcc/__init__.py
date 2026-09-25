"""rcc: receiver-conditioned latent handoffs for sub-agent systems.

Exact KV-cache communication between two instances of one base model. This
module exports the public API; scoring lives at `rcc.harness.score`.
"""

from rcc.cache import Axis, KVCache
from rcc.harness.handoff import handoff
from rcc.meters import FactoredRatio, Meter, Meters, factored, measure
from rcc.pipeline import MeterViolation, Pipeline
from rcc.rope import RopeParams
from rcc.transform import Kind, Transform
from rcc.transforms.select import (
    Select,
    grad_scores,
    h2o_scores,
    selfq_scores,
    snapkv_query_scores,
    snapkv_scores,
    span_schedule,
)

__version__ = "0.1.0"

__all__ = [
    "Axis",
    "FactoredRatio",
    "KVCache",
    "Kind",
    "Meter",
    "MeterViolation",
    "Meters",
    "Pipeline",
    "RopeParams",
    "Select",
    "Transform",
    "__version__",
    "factored",
    "grad_scores",
    "h2o_scores",
    "handoff",
    "measure",
    "selfq_scores",
    "snapkv_query_scores",
    "snapkv_scores",
    "span_schedule",
]
