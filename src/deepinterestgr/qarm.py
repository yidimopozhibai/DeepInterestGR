"""Quality-Aware Reinforcement Mechanism (QARM) equations."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Iterable, Sequence

from .api import CompletionAdapter
from .schemas import BinaryLabel, ItemEvidence
from .sid import RQ_LEVELS, SIDCatalog, SemanticID


ALPHA = 0.5
ADVANTAGE_EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """Auditable decomposition of manuscript reward terms."""

    exact: float
    valid: float
    lcp: float
    quality: float
    alpha: float
    total: float
    decoded_item_id: str | None
    target_item_id: str
    lcp_length: int
    candidate_valid: bool


@dataclass(frozen=True, slots=True)
class GRPOSurrogateBreakdown:
    """Per-sample values used by the pure clipped GRPO objective."""

    ratios: tuple[float, ...]
    clipped_ratios: tuple[float, ...]
    unclipped_terms: tuple[float, ...]
    clipped_terms: tuple[float, ...]
    selected_terms: tuple[float, ...]
    policy_mean: float
    kl_penalty: float
    objective: float


class QARMJudge:
    """Judge one descriptor against source evidence and return only 0 or 1."""

    _SYSTEM_PROMPT = """Judge one item-side descriptor against only the supplied title,
description, and image caption. Return integer 1 when it is specific, actionable, and
grounded in that supplied content; otherwise return integer 0 when vague, generic,
unsupported, or contradictory. Do not assess personalized relevance or infer psychology.
Return the single JSON integer 0 or 1, with no object, explanation, or other text."""

    def __init__(self, adapter: CompletionAdapter) -> None:
        self.adapter = adapter

    def label(
        self,
        *,
        descriptor: str,
        title: str,
        description: str,
        caption: str,
    ) -> int:
        if not isinstance(descriptor, str) or not descriptor.strip():
            raise ValueError("descriptor must be a non-empty string")
        evidence = ItemEvidence(
            title=title, description=description, caption=caption
        )
        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "descriptor": descriptor.strip(),
                        "evidence": evidence.to_dict(),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ]
        raw = self.adapter.complete(messages, temperature=0.0)
        return BinaryLabel.from_json(raw).value


def aggregate_quality(
    labels: Sequence[int], confidences: Sequence[float | None]
) -> float:
    """Compute confidence-weighted q_i with the manuscript fallback rule.

    If all confidences are missing, every weight becomes one. A mixture of
    present and missing confidences is rejected instead of silently imputing.
    """

    labels = tuple(labels)
    confidences = tuple(confidences)
    if not labels:
        raise ValueError("at least one descriptor label is required")
    if len(labels) != len(confidences):
        raise ValueError("labels and confidences must have equal lengths")
    if any(type(label) is not int or label not in (0, 1) for label in labels):
        raise ValueError("labels must contain only integer 0 or 1 values")

    missing = tuple(confidence is None for confidence in confidences)
    if all(missing):
        weights = (1.0,) * len(labels)
    elif any(missing):
        raise ValueError("confidences must be either all present or all missing")
    else:
        parsed: list[float] = []
        for confidence in confidences:
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                raise ValueError("confidence weights must be finite numbers")
            weight = float(confidence)
            if not math.isfinite(weight):
                raise ValueError("confidence weights must be finite")
            if weight < 0.0 or weight > 1.0:
                raise ValueError("confidence weights must be in [0, 1]")
            parsed.append(weight)
        weights = tuple(parsed)

    denominator = math.fsum(weights)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("confidence-weight denominator must be finite and positive")
    numerator = math.fsum(weight * label for weight, label in zip(weights, labels))
    quality = numerator / denominator
    if not math.isfinite(quality):
        raise ValueError("aggregated quality must be finite")
    return quality


def longest_common_rq_prefix(candidate: SemanticID, target: SemanticID) -> int:
    """LCP over exactly the three RQ codes; suffixes are ignored."""

    count = 0
    for candidate_code, target_code in zip(candidate.rq_codes, target.rq_codes):
        if candidate_code != target_code:
            break
        count += 1
    return count


def _generated_rq_prefix(candidate: object, target: SemanticID) -> int:
    """LCP over up to three generated RQ tokens, ignoring any later tokens."""

    if isinstance(candidate, SemanticID):
        candidate_codes: Sequence[object] = candidate.rq_codes
    elif isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
        candidate_codes = candidate
    else:
        return 0
    count = 0
    for candidate_code, target_code in zip(candidate_codes, target.rq_codes):
        if not isinstance(candidate_code, str) or candidate_code != target_code:
            break
        count += 1
    return count


def reward_breakdown(
    *,
    candidate: object,
    target_item_id: str,
    catalog: SIDCatalog,
    item_quality: dict[str, float],
    lambda_valid: float,
    lambda_pref: float,
) -> RewardBreakdown:
    """Compute exact, validity, LCP, and exact-target-gated quality rewards."""

    for name, value in (
        ("lambda_valid", lambda_valid),
        ("lambda_pref", lambda_pref),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite number")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    target_sid = catalog.sid_for(target_item_id)
    decoded_item_id = catalog.try_decode(candidate)
    candidate_valid = decoded_item_id is not None
    exact_match = decoded_item_id == target_item_id
    lcp_length = _generated_rq_prefix(candidate, target_sid)

    exact = float(exact_match)
    valid = float(lambda_valid) * float(candidate_valid)
    lcp = float(lambda_pref) * (lcp_length / RQ_LEVELS)

    quality = 0.0
    if exact_match:
        if target_item_id not in item_quality:
            raise KeyError(f"quality missing for exact target item: {target_item_id}")
        raw_quality = item_quality[target_item_id]
        if isinstance(raw_quality, bool) or not isinstance(raw_quality, (int, float)):
            raise ValueError("item quality must be a finite number in [0, 1]")
        quality = float(raw_quality)
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("item quality must be finite and in [0, 1]")

    total = exact + valid + lcp + ALPHA * quality
    return RewardBreakdown(
        exact=exact,
        valid=valid,
        lcp=lcp,
        quality=quality,
        alpha=ALPHA,
        total=total,
        decoded_item_id=decoded_item_id,
        target_item_id=target_item_id,
        lcp_length=lcp_length,
        candidate_valid=candidate_valid,
    )


def normalized_advantages(rewards: Sequence[float]) -> tuple[float, ...]:
    """Compute (r - mean) / (population_std + 1e-6)."""

    rewards = tuple(float(reward) for reward in rewards)
    if not rewards:
        raise ValueError("rewards must not be empty")
    if any(not math.isfinite(reward) for reward in rewards):
        raise ValueError("rewards must be finite")

    mean = math.fsum(rewards) / len(rewards)
    variance = math.fsum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    standard_deviation = math.sqrt(variance)
    denominator = standard_deviation + ADVANTAGE_EPSILON
    return tuple((reward - mean) / denominator for reward in rewards)


def clipped_grpo_surrogate(
    *,
    new_log_probs: Sequence[float],
    behavior_log_probs: Sequence[float],
    advantages: Sequence[float],
    kl_divergence: float,
    clip_epsilon: float,
    kl_beta: float,
) -> GRPOSurrogateBreakdown:
    """Pure scalar form of the manuscript's clipped GRPO surrogate.

    This function performs no optimizer step and mutates no model state. The
    caller supplies per-candidate log probabilities, normalized advantages, an
    already-computed KL divergence, clipping epsilon, and KL coefficient.
    """

    new_values = tuple(float(value) for value in new_log_probs)
    behavior_values = tuple(float(value) for value in behavior_log_probs)
    advantage_values = tuple(float(value) for value in advantages)
    if not new_values:
        raise ValueError("at least one candidate is required")
    if not (len(new_values) == len(behavior_values) == len(advantage_values)):
        raise ValueError("log-probability and advantage lengths must match")
    if any(
        not math.isfinite(value)
        for value in (*new_values, *behavior_values, *advantage_values)
    ):
        raise ValueError("log probabilities and advantages must be finite")
    for name, value, allow_zero in (
        ("clip_epsilon", clip_epsilon, False),
        ("kl_beta", kl_beta, True),
        ("kl_divergence", kl_divergence, True),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite number")
        numeric = float(value)
        if not math.isfinite(numeric) or (numeric < 0.0 if allow_zero else numeric <= 0.0):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be finite and {qualifier}")
    clip_epsilon = float(clip_epsilon)
    kl_beta = float(kl_beta)
    kl_divergence = float(kl_divergence)

    try:
        ratios = tuple(
            math.exp(new_log_probability - behavior_log_probability)
            for new_log_probability, behavior_log_probability in zip(
                new_values, behavior_values
            )
        )
    except OverflowError as exc:
        raise ValueError("likelihood ratios must be finite") from exc
    if any(not math.isfinite(ratio) for ratio in ratios):
        raise ValueError("likelihood ratios must be finite")
    clipped_ratios = tuple(
        min(max(ratio, 1.0 - clip_epsilon), 1.0 + clip_epsilon)
        for ratio in ratios
    )
    unclipped_terms = tuple(
        ratio * advantage for ratio, advantage in zip(ratios, advantage_values)
    )
    clipped_terms = tuple(
        ratio * advantage
        for ratio, advantage in zip(clipped_ratios, advantage_values)
    )
    selected_terms = tuple(
        min(unclipped, clipped)
        for unclipped, clipped in zip(unclipped_terms, clipped_terms)
    )
    policy_mean = math.fsum(selected_terms) / len(selected_terms)
    kl_penalty = kl_beta * kl_divergence
    objective = policy_mean - kl_penalty
    return GRPOSurrogateBreakdown(
        ratios=ratios,
        clipped_ratios=clipped_ratios,
        unclipped_terms=unclipped_terms,
        clipped_terms=clipped_terms,
        selected_terms=selected_terms,
        policy_mean=policy_mean,
        kl_penalty=kl_penalty,
        objective=objective,
    )
