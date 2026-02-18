"""Tests for P1: default_provider on Client (Spec 2.2).

Covers:
- default_provider used when request omits provider
- explicit provider overrides default
- no default + no provider -> ConfigurationError
- from_env() sets first registered as default
- constructor providers dict
- catalog lookup wins over default
- default_provider property read access
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from attractor_llm.client import Client
from attractor_llm.errors import ConfigurationError
from attractor_llm.types import FinishReason, Message, Request, Response, StreamEvent, Usage

# ------------------------------------------------------------------ #
# Minimal mock adapter
# ------------------------------------------------------------------ #


class _MockAdapter:
    """Minimal stub adapter for routing tests."""

    def __init__(self, provider: str, text: str = "ok") -> None:
        self._provider = provider
        self._text = text

    @property
    def provider_name(self) -> str:
        return self._provider

    async def complete(self, request: Request) -> Response:
        return Response(
            id="mock",
            model=request.model,
            provider=self._provider,
            message=Message.assistant(self._text),
            finish_reason=FinishReason.STOP,
            usage=Usage(input_tokens=1, output_tokens=1),
        )

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Stub: immediately-exhausted async generator for smoke tests."""
        return
        yield  # type: ignore[misc]  # unreachable – makes this an async generator

    async def close(self) -> None:
        pass


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #


class TestDefaultProviderRouting:
    @pytest.mark.asyncio
    async def test_default_provider_used_when_request_omits_provider(self):
        """When no provider on request, Client falls back to default_provider."""
        adapter_a = _MockAdapter("alpha", "from-alpha")
        adapter_b = _MockAdapter("beta", "from-beta")
        client = Client(default_provider="beta")
        client.register_adapter("alpha", adapter_a)
        client.register_adapter("beta", adapter_b)

        # "unknown-model-xyz" is not in the catalog -- must use default
        resp = await client.complete(
            Request(model="unknown-model-xyz", messages=[Message.user("hi")])
        )
        assert resp.provider == "beta"
        assert resp.text == "from-beta"

    @pytest.mark.asyncio
    async def test_explicit_provider_overrides_default(self):
        """Explicit request.provider always wins over default_provider."""
        adapter_a = _MockAdapter("alpha", "from-alpha")
        adapter_b = _MockAdapter("beta", "from-beta")
        client = Client(default_provider="beta")
        client.register_adapter("alpha", adapter_a)
        client.register_adapter("beta", adapter_b)

        resp = await client.complete(
            Request(model="anything", provider="alpha", messages=[Message.user("hi")])
        )
        assert resp.provider == "alpha"
        assert resp.text == "from-alpha"

    @pytest.mark.asyncio
    async def test_no_adapters_raises_config_error(self):
        """Client with no adapters at all raises ConfigurationError."""
        client = Client()
        with pytest.raises(ConfigurationError):
            await client.complete(Request(model="unknown", messages=[Message.user("hi")]))

    @pytest.mark.asyncio
    async def test_dangling_default_provider_raises_config_error(self):
        """default_provider set to unregistered name raises clearly."""
        client = Client(default_provider="ghost")
        client.register_adapter("openai", _MockAdapter("openai"))
        with pytest.raises(ConfigurationError, match="ghost"):
            await client.complete(Request(model="unknown", messages=[Message.user("hi")]))

    @pytest.mark.asyncio
    async def test_stream_uses_default_provider_routing(self):
        """stream() also resolves adapter via _resolve_adapter / default_provider."""
        adapter_a = _MockAdapter("alpha")
        adapter_b = _MockAdapter("beta")
        client = Client(default_provider="beta")
        client.register_adapter("alpha", adapter_a)
        client.register_adapter("beta", adapter_b)

        # Should not raise; default_provider routes to "beta"
        stream = await client.stream(
            Request(model="unknown-model-xyz", messages=[Message.user("hi")])
        )
        assert stream is not None

    def test_from_env_sets_first_registered_as_default(self, monkeypatch):
        """from_env() first-found env key becomes default_provider."""
        # Only set OPENAI_API_KEY so it is the sole (and thus first) adapter
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-openai")

        client = Client.from_env()
        assert "openai" in client._adapters
        assert client.default_provider == "openai"

    @pytest.mark.asyncio
    async def test_constructor_with_providers_dict(self):
        """Passing providers= to constructor registers all and sets first as default."""
        adapter_x = _MockAdapter("x", "from-x")
        adapter_y = _MockAdapter("y", "from-y")

        client = Client(providers={"x": adapter_x, "y": adapter_y})

        assert "x" in client._adapters
        assert "y" in client._adapters
        # First in dict becomes default
        assert client.default_provider == "x"

        # Route unknown model through default (x)
        resp = await client.complete(
            Request(model="no-catalog-match", messages=[Message.user("hi")])
        )
        assert resp.provider == "x"

    @pytest.mark.asyncio
    async def test_catalog_lookup_before_default(self):
        """Catalog-identified provider wins over default_provider (step 2 > step 3)."""
        anthropic_adapter = _MockAdapter("anthropic", "from-anthropic")
        openai_adapter = _MockAdapter("openai", "from-openai")

        # default is openai, but the model is catalogued under anthropic
        client = Client(default_provider="openai")
        client.register_adapter("anthropic", anthropic_adapter)
        client.register_adapter("openai", openai_adapter)

        # claude-sonnet-4-5 is in the catalog -> provider = "anthropic"
        resp = await client.complete(
            Request(model="claude-sonnet-4-5", messages=[Message.user("hi")])
        )
        assert resp.provider == "anthropic"
        assert resp.text == "from-anthropic"

    def test_default_provider_property(self):
        """default_provider property returns the current value."""
        client = Client()
        assert client.default_provider is None

        client.register_adapter("first", _MockAdapter("first"))
        assert client.default_provider == "first"

        # Registering more adapters does not change the already-set default
        client.register_adapter("second", _MockAdapter("second"))
        assert client.default_provider == "first"

    def test_constructor_explicit_default_not_overwritten_by_register(self):
        """Explicit default_provider= in constructor is preserved on first register_adapter."""
        adapter_a = _MockAdapter("a")
        adapter_b = _MockAdapter("b")

        client = Client(default_provider="b")
        client.register_adapter("a", adapter_a)  # first registered, but default already set
        client.register_adapter("b", adapter_b)

        assert client.default_provider == "b"
