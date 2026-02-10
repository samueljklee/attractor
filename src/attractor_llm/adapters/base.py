"""Base adapter protocol and configuration for provider adapters.

Defines the contract that all provider adapters must implement,
and shared configuration types.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from attractor_llm.retry import RetryPolicy
from attractor_llm.types import Request, Response, StreamEvent


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for a provider adapter.

    Each adapter receives this at construction time. API keys,
    base URLs, and timeouts are provider-specific but share this shape.
    """

    api_key: str
    base_url: str | None = None
    timeout: float = 60.0
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    default_headers: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class ProviderAdapter(Protocol):
    """Protocol that all provider adapters must implement. Spec §7.1.

    Each adapter translates unified Request/Response types to/from
    the provider's native API format.
    """

    @property
    def provider_name(self) -> str:
        """The provider identifier (e.g., 'anthropic', 'openai', 'gemini')."""
        ...

    async def complete(self, request: Request) -> Response:
        """Send a request and return the complete response. Spec §4.1.

        Blocks until the model finishes generating.
        """
        ...

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Send a request and return a streaming event iterator. Spec §4.2.

        Yields StreamEvent objects as the model generates.
        The first event should be START with model/provider metadata.
        The last event should be FINISH with the finish reason.
        """
        ...

    async def close(self) -> None:
        """Release resources (HTTP connections, etc.)."""
        ...
