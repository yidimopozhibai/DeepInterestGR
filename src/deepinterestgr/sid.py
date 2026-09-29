"""Semantic-ID catalog with exactly three RQ codes plus optional suffix."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence


RQ_LEVELS = 3


def _validate_token(token: object, *, field: str) -> str:
    if not isinstance(token, str) or not token.strip():
        raise ValueError(f"{field} must be a non-empty string token")
    if token != token.strip():
        raise ValueError(f"{field} must not have surrounding whitespace")
    return token


@dataclass(frozen=True, slots=True)
class SemanticID:
    """Three hierarchical RQ tokens and an optional collision-only suffix."""

    rq_codes: tuple[str, str, str]
    collision_suffix: str | None = None

    def __post_init__(self) -> None:
        codes = tuple(self.rq_codes)
        if len(codes) != RQ_LEVELS:
            raise ValueError(f"rq_codes must contain exactly {RQ_LEVELS} tokens")
        codes = tuple(
            _validate_token(code, field=f"rq_codes[{index}]")
            for index, code in enumerate(codes)
        )
        suffix = self.collision_suffix
        if suffix is not None:
            suffix = _validate_token(suffix, field="collision_suffix")
        object.__setattr__(self, "rq_codes", codes)
        object.__setattr__(self, "collision_suffix", suffix)

    @classmethod
    def from_tokens(cls, tokens: Sequence[str]) -> "SemanticID":
        if isinstance(tokens, (str, bytes)):
            raise ValueError("SID tokens must be a sequence of token strings")
        values = tuple(tokens)
        if len(values) == RQ_LEVELS:
            return cls(rq_codes=values)  # type: ignore[arg-type]
        if len(values) == RQ_LEVELS + 1:
            return cls(
                rq_codes=values[:RQ_LEVELS],  # type: ignore[arg-type]
                collision_suffix=values[RQ_LEVELS],
            )
        raise ValueError("SID must contain three RQ codes and at most one suffix")

    @property
    def tokens(self) -> tuple[str, ...]:
        if self.collision_suffix is None:
            return self.rq_codes
        return (*self.rq_codes, self.collision_suffix)


class SIDCatalog:
    """Validated one-to-one item/SID decoding table.

    A suffix is permitted exactly when the item's three-code tuple collides with
    another catalog item. Every item in a collision bucket must have a unique
    suffix; non-colliding items must not have one.
    """

    def __init__(self, item_to_sid: Mapping[str, SemanticID]) -> None:
        if not item_to_sid:
            raise ValueError("catalog must not be empty")
        copied: dict[str, SemanticID] = {}
        buckets: dict[tuple[str, str, str], list[tuple[str, SemanticID]]] = {}
        for item_id, sid in item_to_sid.items():
            item_id = _validate_token(item_id, field="item_id")
            if not isinstance(sid, SemanticID):
                raise TypeError("catalog values must be SemanticID instances")
            if item_id in copied:
                raise ValueError(f"duplicate item_id: {item_id}")
            copied[item_id] = sid
            buckets.setdefault(sid.rq_codes, []).append((item_id, sid))

        token_to_item: dict[tuple[str, ...], str] = {}
        for bucket in buckets.values():
            collided = len(bucket) > 1
            suffixes: set[str] = set()
            for item_id, sid in bucket:
                if collided and sid.collision_suffix is None:
                    raise ValueError(
                        "every item sharing three RQ codes needs a collision suffix"
                    )
                if not collided and sid.collision_suffix is not None:
                    raise ValueError(
                        "a collision suffix is forbidden for a non-colliding SID"
                    )
                if sid.collision_suffix is not None:
                    if sid.collision_suffix in suffixes:
                        raise ValueError("collision suffixes must be unique within a bucket")
                    suffixes.add(sid.collision_suffix)
                if sid.tokens in token_to_item:
                    raise ValueError("full SID sequences must be globally unique")
                token_to_item[sid.tokens] = item_id

        self._item_to_sid = MappingProxyType(copied)
        self._token_to_item = MappingProxyType(token_to_item)

    @property
    def item_to_sid(self) -> Mapping[str, SemanticID]:
        return self._item_to_sid

    def sid_for(self, item_id: str) -> SemanticID:
        try:
            return self._item_to_sid[item_id]
        except KeyError as exc:
            raise KeyError(f"unknown item_id: {item_id}") from exc

    def decode(self, candidate: SemanticID | Sequence[str]) -> str | None:
        """Decode a well-formed SID, returning ``None`` when it is not in the catalog."""

        sid = candidate if isinstance(candidate, SemanticID) else SemanticID.from_tokens(candidate)
        return self._token_to_item.get(sid.tokens)

    def try_decode(self, candidate: object) -> str | None:
        """Decode generated output without raising for malformed SID sequences."""

        if not isinstance(candidate, SemanticID) and (
            not isinstance(candidate, Sequence)
            or isinstance(candidate, (str, bytes))
        ):
            return None
        try:
            return self.decode(candidate)
        except (TypeError, ValueError):
            return None

    def is_valid(self, candidate: object) -> bool:
        return self.try_decode(candidate) is not None
