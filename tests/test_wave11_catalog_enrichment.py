"""Tests for Wave 11 P2 – catalog enrichment (§2.9).

Covers:
  - New ModelInfo fields: input_cost_per_million, output_cost_per_million, aliases
  - Alias population for all catalog entries
  - get_model_info alias resolution (with exact-ID precedence)
  - get_latest_model per provider and filtered by capability
"""
from __future__ import annotations

import pytest

from attractor_llm.catalog import (
    MODEL_CATALOG,
    ModelInfo,
    get_latest_model,
    get_model_info,
)


# ---------------------------------------------------------------------------
# ModelInfo field presence
# ---------------------------------------------------------------------------


def test_model_info_has_cost_fields() -> None:
    """input_cost_per_million and output_cost_per_million exist and default to None."""
    m = ModelInfo(
        id="test-model",
        provider="test",
        display_name="Test Model",
        context_window=8_000,
    )
    assert hasattr(m, "input_cost_per_million")
    assert hasattr(m, "output_cost_per_million")
    assert m.input_cost_per_million is None
    assert m.output_cost_per_million is None


def test_model_info_has_aliases_field() -> None:
    """aliases exists and defaults to an empty list."""
    m = ModelInfo(
        id="test-model",
        provider="test",
        display_name="Test Model",
        context_window=8_000,
    )
    assert hasattr(m, "aliases")
    assert isinstance(m.aliases, list)
    assert m.aliases == []


# ---------------------------------------------------------------------------
# Alias population
# ---------------------------------------------------------------------------


def test_all_catalog_entries_have_aliases() -> None:
    """Every entry in MODEL_CATALOG has at least one alias."""
    assert MODEL_CATALOG, "Catalog must not be empty"
    for entry in MODEL_CATALOG:
        assert len(entry.aliases) >= 1, (
            f"Model {entry.id!r} has no aliases – all catalog entries must have at least one"
        )


# ---------------------------------------------------------------------------
# get_model_info: alias resolution
# ---------------------------------------------------------------------------


def test_alias_resolution_in_get_model_info() -> None:
    """get_model_info('sonnet') resolves to the claude-sonnet-4-5 entry."""
    result = get_model_info("sonnet")
    assert result is not None
    assert result.id == "claude-sonnet-4-5"


def test_alias_resolution_opus() -> None:
    """get_model_info('opus') resolves to the claude-opus-4-6 entry."""
    result = get_model_info("opus")
    assert result is not None
    assert result.id == "claude-opus-4-6"


def test_alias_resolution_flash() -> None:
    """get_model_info('flash') resolves to the gemini-3-flash-preview entry."""
    result = get_model_info("flash")
    assert result is not None
    assert result.id == "gemini-3-flash-preview"


def test_exact_id_takes_precedence_over_alias() -> None:
    """Exact ID lookup wins over alias lookup.

    'gpt-5.2-mini' is a real catalog ID; it must be returned directly
    (not confused with an alias match on a different entry).
    """
    result = get_model_info("gpt-5.2-mini")
    assert result is not None
    assert result.id == "gpt-5.2-mini"


def test_exact_id_lookup_still_works() -> None:
    """Full model IDs continue to resolve correctly after alias support was added."""
    for entry in MODEL_CATALOG:
        found = get_model_info(entry.id)
        assert found is not None, f"Exact lookup failed for {entry.id!r}"
        assert found.id == entry.id


def test_unknown_alias_returns_none() -> None:
    """An unrecognised string that is neither an ID nor an alias returns None."""
    assert get_model_info("nonexistent") is None
    assert get_model_info("") is None
    assert get_model_info("totally-made-up-model-xyz") is None


# ---------------------------------------------------------------------------
# get_latest_model: basic provider queries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini"])
def test_get_latest_model_per_provider(provider: str) -> None:
    """get_latest_model returns a ModelInfo for each known provider."""
    result = get_latest_model(provider)
    assert result is not None, f"Expected a model for provider {provider!r}"
    assert isinstance(result, ModelInfo)
    assert result.provider == provider


def test_get_latest_model_returns_first_catalog_entry_for_provider() -> None:
    """The returned model is the first entry in catalog order for that provider."""
    for provider in ("anthropic", "openai", "gemini"):
        expected_first = next(m for m in MODEL_CATALOG if m.provider == provider)
        result = get_latest_model(provider)
        assert result is not None
        assert result.id == expected_first.id, (
            f"For provider {provider!r}, expected first-catalog entry {expected_first.id!r}, "
            f"got {result.id!r}"
        )


# ---------------------------------------------------------------------------
# get_latest_model: capability filtering
# ---------------------------------------------------------------------------


def test_get_latest_model_with_capability_tools() -> None:
    """Filtering by 'tools' returns a model that has supports_tools=True."""
    for provider in ("anthropic", "openai", "gemini"):
        result = get_latest_model(provider, "tools")
        assert result is not None, f"No tools-capable model found for {provider!r}"
        assert result.supports_tools is True
        assert result.provider == provider


def test_get_latest_model_with_capability_vision() -> None:
    """Filtering by 'vision' returns a model that has supports_vision=True."""
    for provider in ("anthropic", "openai", "gemini"):
        result = get_latest_model(provider, "vision")
        assert result is not None, f"No vision-capable model found for {provider!r}"
        assert result.supports_vision is True
        assert result.provider == provider


def test_get_latest_model_with_capability_reasoning() -> None:
    """Filtering by 'reasoning' returns a model that has supports_reasoning=True."""
    for provider in ("anthropic", "openai", "gemini"):
        result = get_latest_model(provider, "reasoning")
        assert result is not None, f"No reasoning-capable model found for {provider!r}"
        assert result.supports_reasoning is True
        assert result.provider == provider


# ---------------------------------------------------------------------------
# get_latest_model: None cases
# ---------------------------------------------------------------------------


def test_get_latest_model_unknown_provider_returns_none() -> None:
    """An unrecognised provider returns None."""
    assert get_latest_model("unknown-provider") is None
    assert get_latest_model("") is None
    assert get_latest_model("cohere") is None


def test_get_latest_model_unsatisfiable_capability_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returns None when no model for the given provider has the requested capability."""
    # Build a fake catalog with one anthropic model that has no reasoning support
    fake_catalog = [
        ModelInfo(
            id="limited-model",
            provider="anthropic",
            display_name="Limited Model",
            context_window=100_000,
            supports_tools=True,
            supports_vision=True,
            supports_reasoning=False,  # <-- no reasoning
            aliases=["limited"],
        )
    ]
    monkeypatch.setattr("attractor_llm.catalog.MODEL_CATALOG", fake_catalog)

    result = get_latest_model("anthropic", "reasoning")
    assert result is None


def test_get_latest_model_unknown_capability_returns_none() -> None:
    """An unknown capability string always returns None (no field to filter on)."""
    assert get_latest_model("anthropic", "telekinesis") is None
    assert get_latest_model("openai", "audio") is None


# ---------------------------------------------------------------------------
# Export surface
# ---------------------------------------------------------------------------


def test_get_latest_model_exported_from_package() -> None:
    """get_latest_model is accessible from the top-level attractor_llm package."""
    import attractor_llm

    assert hasattr(attractor_llm, "get_latest_model")
    assert callable(attractor_llm.get_latest_model)
    assert "get_latest_model" in attractor_llm.__all__
