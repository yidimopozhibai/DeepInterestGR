"""Strict, dependency-free schemas used by the reference implementation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
import unicodedata
from typing import Any, Mapping


class SchemaValidationError(ValueError):
    """Raised when an input or model response violates an exact schema."""


def _reject_json_constant(token: str) -> None:
    raise SchemaValidationError(f"non-standard JSON constant is not allowed: {token}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SchemaValidationError(f"duplicate JSON key is not allowed: {key}")
        result[key] = value
    return result


def loads_strict_json(payload: str) -> Any:
    """Parse JSON while rejecting NaN, infinities, and duplicate object keys."""

    if not isinstance(payload, str):
        raise SchemaValidationError("JSON payload must be a string")
    try:
        return json.loads(
            payload,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except SchemaValidationError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise SchemaValidationError(f"invalid JSON: {exc}") from exc


def _require_exact_keys(
    value: Any, required: set[str], *, context: str
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise SchemaValidationError(f"{context} must be a JSON object")
    keys = set(value)
    missing = required - keys
    extra = keys - required
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if extra:
            details.append(f"extra={sorted(extra)}")
        raise SchemaValidationError(f"{context} has invalid fields ({', '.join(details)})")
    return value


def _require_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise SchemaValidationError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise SchemaValidationError(f"{field} must not be empty")
    return normalized


def _require_confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError("confidence must be a JSON number")
    confidence = float(value)
    if not math.isfinite(confidence):
        raise SchemaValidationError("confidence must be finite")
    if not 0.0 <= confidence <= 1.0:
        raise SchemaValidationError("confidence must be in [0, 1]")
    return confidence


def normalize_descriptor(value: str) -> str:
    """Normalize descriptor text for deterministic duplicate detection."""

    value = unicodedata.normalize("NFKC", value).casefold().strip()
    return re.sub(r"\s+", " ", value)


@dataclass(frozen=True, slots=True)
class ItemEvidence:
    """The three item-side evidence fields consumed by DCIM and QARM."""

    title: str
    description: str
    caption: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "title", _require_text(self.title, field="title"))
        object.__setattr__(
            self, "description", _require_text(self.description, field="description")
        )
        object.__setattr__(self, "caption", _require_text(self.caption, field="caption"))

    @classmethod
    def from_mapping(cls, value: Any) -> "ItemEvidence":
        obj = _require_exact_keys(
            value, {"title", "description", "caption"}, context="item evidence"
        )
        return cls(
            title=obj["title"],
            description=obj["description"],
            caption=obj["caption"],
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "description": self.description,
            "caption": self.caption,
        }


@dataclass(frozen=True, slots=True)
class GroundedDescriptor:
    """An ordered item-side descriptor and its DCIM confidence."""

    descriptor: str
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "descriptor", _require_text(self.descriptor, field="descriptor")
        )
        object.__setattr__(self, "confidence", _require_confidence(self.confidence))

    @classmethod
    def from_mapping(cls, value: Any) -> "GroundedDescriptor":
        obj = _require_exact_keys(
            value, {"descriptor", "confidence"}, context="descriptor"
        )
        return cls(descriptor=obj["descriptor"], confidence=obj["confidence"])

    def to_dict(self) -> dict[str, object]:
        return {"descriptor": self.descriptor, "confidence": self.confidence}


@dataclass(frozen=True, slots=True)
class DescriptorResponse:
    """A non-empty ordered collection of unique grounded descriptors."""

    descriptors: tuple[GroundedDescriptor, ...]

    def __post_init__(self) -> None:
        descriptors = tuple(self.descriptors)
        if not descriptors:
            raise SchemaValidationError("descriptors must be a non-empty array")
        if not all(isinstance(value, GroundedDescriptor) for value in descriptors):
            raise SchemaValidationError("descriptors contains an invalid value")
        normalized = [normalize_descriptor(value.descriptor) for value in descriptors]
        if len(normalized) != len(set(normalized)):
            raise SchemaValidationError("duplicate normalized descriptors are not allowed")
        object.__setattr__(self, "descriptors", descriptors)

    @classmethod
    def from_mapping(cls, value: Any) -> "DescriptorResponse":
        obj = _require_exact_keys(value, {"descriptors"}, context="DCIM response")
        raw_descriptors = obj["descriptors"]
        if not isinstance(raw_descriptors, list):
            raise SchemaValidationError("descriptors must be a JSON array")
        return cls(
            descriptors=tuple(
                GroundedDescriptor.from_mapping(descriptor)
                for descriptor in raw_descriptors
            )
        )

    @classmethod
    def from_json(cls, payload: str) -> "DescriptorResponse":
        return cls.from_mapping(loads_strict_json(payload))

    def to_dict(self) -> dict[str, list[dict[str, object]]]:
        return {"descriptors": [value.to_dict() for value in self.descriptors]}


@dataclass(frozen=True, slots=True)
class BinaryLabel:
    """A strict integer label; booleans are deliberately not integers here."""

    value: int

    def __post_init__(self) -> None:
        if type(self.value) is not int or self.value not in (0, 1):
            raise SchemaValidationError("label must be the integer 0 or 1")

    @classmethod
    def from_json(cls, payload: str) -> "BinaryLabel":
        return cls(loads_strict_json(payload))


DCIM_RESPONSE_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["descriptors"],
    "properties": {
        "descriptors": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["descriptor", "confidence"],
                "properties": {
                    "descriptor": {"type": "string", "minLength": 1},
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
            },
        }
    },
}
