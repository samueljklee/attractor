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
from attractor_llm.errors import ConfigurationError, SDKError
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

    @classmethod
    def from_env(cls, **kwargs: Any) -> Client:
        """Create a Client with providers auto-detected from environment variables.

        Checks for standard API key env vars and registers the corresponding
        adapter for each one found. First registered adapter is the default.

        Supported env vars (Spec §2.2):
        - OPENAI_API_KEY → OpenAI adapter
        - ANTHROPIC_API_KEY → Anthropic adapter
        - GEMINI_API_KEY or GOOGLE_API_KEY → Gemini adapter

        Args:
            **kwargs: Passed to Client.__init__ (e.g., retry_policy).

        Returns:
            A configured Client instance.
        """
        import os

        from attractor_llm.adapters.base import ProviderConfig

        client = cls(**kwargs)

        if api_key := os.environ.get("OPENAI_API_KEY"):
            from attractor_llm.adapters.openai import OpenAIAdapter

            client.register_adapter("openai", OpenAIAdapter(ProviderConfig(api_key=api_key)))

        if api_key := os.environ.get("ANTHROPIC_API_KEY"):
            from attractor_llm.adapters.anthropic import AnthropicAdapter

            client.register_adapter("anthropic", AnthropicAdapter(ProviderConfig(api_key=api_key)))

        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if gemini_key:
            from attractor_llm.adapters.gemini import GeminiAdapter

            client.register_adapter("gemini", GeminiAdapter(ProviderConfig(api_key=gemini_key)))

        return client

    def _resolve_adapter(self, request: Request) -> ProviderAdapter:
        """Resolve which adapter to use for a request.

        Resolution order:
        1. Explicit ``request.provider`` field
        2. Model catalog lookup (model ID -> provider)
        3. Fail with ConfigurationError

        Raises:
            ConfigurationError: If no adapter can be resolved.
        """
        # 1. Explicit provider
        if request.provider:
            adapter = self._adapters.get(request.provider)
            if adapter:
                return adapter
            raise ConfigurationError(
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

        raise ConfigurationError(
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


# ------------------------------------------------------------------ #
# Module-level default client (Spec §2.2)
# ------------------------------------------------------------------ #

_default_client: Client | None = None


def set_default_client(client: Client) -> None:
    """Set the module-level default client. Spec §2.2."""
    global _default_client
    _default_client = client


def get_default_client() -> Client:
    """Get the module-level default client. Spec §2.2.

    Raises:
        ConfigurationError: If no default client has been set.
    """
    if _default_client is None:
        raise ConfigurationError(
            "No default client configured. "
            "Call set_default_client() or Client.from_env() first."
        )
    return _default_client
