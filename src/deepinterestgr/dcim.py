"""Deep Contextual Interest Mining (DCIM)."""

from __future__ import annotations

import json
from typing import Any

from .api import CompletionAdapter
from .schemas import (
    DCIM_RESPONSE_JSON_SCHEMA,
    DescriptorResponse,
    ItemEvidence,
)


TEXT_FIELD_SEPARATOR = "\n"


_DCIM_SYSTEM_PROMPT = """You mine ordered deep-interest descriptors: latent usage or purchase
motivations that an item may serve even when users have not explicitly articulated them. Infer
these implicit contexts using only the supplied title, description, and image caption. Treat every
descriptor as an item-side grounded proxy, not a verified psychological state or a personalized
claim about a particular user. Each descriptor must be specific, actionable, and supported by the
supplied evidence. Do not infer personality, emotion, demographics, identity, or private traits.
Return only the requested strict JSON object, ordered from strongest to weakest evidence, with no
prose or fallback."""


class DCIMExtractor:
    """Extract strict, ordered, content-grounded item-side descriptors."""

    def __init__(self, adapter: CompletionAdapter) -> None:
        self.adapter = adapter

    def extract(
        self, *, title: str, description: str, caption: str
    ) -> DescriptorResponse:
        evidence = ItemEvidence(
            title=title, description=description, caption=caption
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _DCIM_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Extract the grounded descriptors from this JSON evidence:\n"
                    + json.dumps(evidence.to_dict(), ensure_ascii=False, sort_keys=True)
                ),
            },
        ]
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "grounded_item_descriptors",
                "strict": True,
                "schema": DCIM_RESPONSE_JSON_SCHEMA,
            },
        }
        raw = self.adapter.complete(
            messages, response_format=response_format, temperature=0.0
        )
        # Parsing failures are surfaced. There is deliberately no generated fallback.
        return DescriptorResponse.from_json(raw)


def build_enriched_text(
    *,
    title: str,
    description: str,
    caption: str,
    descriptors: DescriptorResponse,
) -> str:
    """Construct x_i^text in a deterministic evidence order.

    The sequence is title, description, visual caption, then descriptor text in
    the order returned by DCIM. Confidence values remain QARM metadata and are intentionally excluded from
    the text passed to the external Qwen3-Embedding-4B integration boundary.
    """

    evidence = ItemEvidence(
        title=title, description=description, caption=caption
    )
    if not isinstance(descriptors, DescriptorResponse):
        raise TypeError("descriptors must be a DescriptorResponse")
    fields = [
        evidence.title,
        evidence.description,
        evidence.caption,
        *(value.descriptor for value in descriptors.descriptors),
    ]
    return TEXT_FIELD_SEPARATOR.join(fields)
