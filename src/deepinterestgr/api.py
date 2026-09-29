"""Injectable adapter for OpenAI-compatible chat-completion APIs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol


Message = Mapping[str, Any]


class CompletionAdapter(Protocol):
    """Small interface used by captioning, DCIM, and QARM."""

    def complete(
        self,
        messages: Sequence[Message],
        *,
        response_format: Mapping[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Return the assistant's textual content."""


class OpenAICompatibleAdapter:
    """Chat-completion adapter with explicit, externally supplied connection data.

    The optional ``client`` injection is useful for tests and compatible SDK
    wrappers. If no client is supplied, the ``openai`` package is imported only
    when the first request is made.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        deployment: str,
        client: Any | None = None,
    ) -> None:
        self.api_key = self._required_string(api_key, "api_key")
        self.base_url = self._required_string(base_url, "base_url")
        self.deployment = self._required_string(deployment, "deployment")
        self._client = client

    @staticmethod
    def _required_string(value: object, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value.strip()

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "Install the optional 'api' dependencies to use the default client"
                ) from exc
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def complete(
        self,
        messages: Sequence[Message],
        *,
        response_format: Mapping[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> str:
        if not messages:
            raise ValueError("messages must not be empty")
        request: dict[str, Any] = {
            "model": self.deployment,
            "messages": list(messages),
            "temperature": temperature,
        }
        if response_format is not None:
            request["response_format"] = dict(response_format)
        response = self._get_client().chat.completions.create(**request)
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise RuntimeError("API response has no assistant text content") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("API response has empty assistant text content")
        return content.strip()
