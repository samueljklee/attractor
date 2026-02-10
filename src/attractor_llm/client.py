"""Top-level LLM Client with provider routing and middleware.

The Client is the main entry point for the SDK. It routes requests
to the appropriate provider adapter based on the model string or
explicit provider field, and applies middleware (logging, caching, etc.).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from attractor_llm.adapters.base import ProviderAdapter
from attractor_llm.catalog import get_model_info
from attractor_llm.errors import InvalidRequestError, SDKError
from attractor_llm.retry import RetryPolicy, retry_with_policy
from attractor_llm.types import (
    Request,
    Response,
    StreamEvent,
)

# Middleware type: receives request and next handler, returns response
Middleware = Any  # TODO: define proper middleware protocol


class Client:
    """Unified LLM Client with provider routing. Spec §2.1-2.6.

    Usage::

        client = Client()
        client.register_adapter("anthropic", AnthropicAdapter(config))
        client.register_adapter("openai", OpenAIAdapter(config))

        response = await client.complete(Request(
            model="claude-sonnet-4-5",
            messages=[Message.user("Hello")],
        ))
    """

    def __init__(
        self,
        *,
        retry_policy: RetryPolicy | None = None,
        middleware: list[Middleware] | None = None,
    ) -> None:
        self._adapters: dict[str, ProviderAdapter] = {}
        self._retry_policy = retry_policy or RetryPolicy()
        self._middleware = middleware or []

    def register_adapter(self, provider: str, adapter: ProviderAdapter) -> None:
        """Register a provider adapter.

        Args:
            provider: Provider name (e.g., "anthropic", "openai", "gemini").
            adapter: The adapter instance implementing ProviderAdapter.
        """
        self._adapters[provider] = adapter

    def _resolve_adapter(self, request: Request) -> ProviderAdapter:
        """Resolve which adapter to use for a request.

        Resolution order:
        1. Explicit ``request.provider`` field
        2. Model catalog lookup (model ID -> provider)
        3. Fail with InvalidRequestError

        Raises:
            InvalidRequestError: If no adapter can be resolved.
        """
        # 1. Explicit provider
        if request.provider:
            adapter = self._adapters.get(request.provider)
            if adapter:
                return adapter
            raise InvalidRequestError(
                f"Provider {request.provider!r} not registered. "
                f"Available: {list(self._adapters.keys())}"
            )

        # 2. Catalog lookup
        model_info = get_model_info(request.model)
        if model_info:
            adapter = self._adapters.get(model_info.provider)
            if adapter:
                return adapter

        # 3. Try to infer from model string prefix heuristics
        model_lower = request.model.lower()
        for provider_name, adapter in self._adapters.items():
            if provider_name in model_lower or model_lower.startswith(
                ("claude", "gpt", "gemini", "o1", "o3", "o4")
            ):
                # More specific matching
                if model_lower.startswith("claude") and provider_name == "anthropic":
                    return adapter
                if model_lower.startswith(("gpt", "o1", "o3", "o4")) and provider_name == "openai":
                    return adapter
                if model_lower.startswith("gemini") and provider_name == "gemini":
                    return adapter

        raise InvalidRequestError(
            f"Cannot resolve provider for model {request.model!r}. "
            f"Set request.provider explicitly or register the provider. "
            f"Available: {list(self._adapters.keys())}"
        )

    async def complete(self, request: Request) -> Response:
        """Send a request and return the complete response. Spec §4.1.

        Routes to the appropriate adapter, applies retry policy.
        """
        adapter = self._resolve_adapter(request)

        async def _do_complete() -> Response:
            return await adapter.complete(request)

        return await retry_with_policy(_do_complete, self._retry_policy)

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Send a request and return a streaming event iterator. Spec §4.2.

        Note: Streaming does not retry mid-stream. If the stream fails
        after partial data has been delivered, a StreamError is raised.
        Retry only applies to the initial connection.
        """
        adapter = self._resolve_adapter(request)
        return await adapter.stream(request)  # type: ignore[return-value]

    async def close(self) -> None:
        """Close all registered adapters and release resources."""
        errors: list[Exception] = []
        for adapter in self._adapters.values():
            try:
                await adapter.close()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        if errors:
            raise SDKError(f"Errors closing adapters: {errors}")

    async def __aenter__(self) -> Client:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()
