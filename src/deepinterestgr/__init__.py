"""DeepInterestGR core implementation of CMSA, DCIM, and QARM."""

from .api import CompletionAdapter, OpenAICompatibleAdapter
from .cmsa import CLIP_MODEL_ID, CMSACaptioner, CMSAFusion
from .dcim import DCIMExtractor, build_enriched_text
from .qarm import (
    ADVANTAGE_EPSILON,
    ALPHA,
    GRPOSurrogateBreakdown,
    QARMJudge,
    RewardBreakdown,
    aggregate_quality,
    clipped_grpo_surrogate,
    normalized_advantages,
    reward_breakdown,
)
from .schemas import (
    BinaryLabel,
    DescriptorResponse,
    GroundedDescriptor,
    ItemEvidence,
    SchemaValidationError,
)
from .sid import RQ_LEVELS, SIDCatalog, SemanticID

__all__ = [
    "ADVANTAGE_EPSILON",
    "ALPHA",
    "BinaryLabel",
    "CLIP_MODEL_ID",
    "CMSACaptioner",
    "CMSAFusion",
    "CompletionAdapter",
    "DCIMExtractor",
    "DescriptorResponse",
    "GRPOSurrogateBreakdown",
    "GroundedDescriptor",
    "ItemEvidence",
    "OpenAICompatibleAdapter",
    "QARMJudge",
    "RQ_LEVELS",
    "RewardBreakdown",
    "SIDCatalog",
    "SchemaValidationError",
    "SemanticID",
    "aggregate_quality",
    "build_enriched_text",
    "clipped_grpo_surrogate",
    "normalized_advantages",
    "reward_breakdown",
]

__version__ = "0.1.0"
