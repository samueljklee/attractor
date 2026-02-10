from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    """Metadata for a known LLM model."""

    id: str
    provider: str
    display_name: str
    context_window: int
    max_output: int | None = None
    supports_tools: bool = False
    supports_vision: bool = False
    supports_reasoning: bool = False


MODEL_CATALOG: list[ModelInfo] = [
    ModelInfo(
        id="claude-opus-4-6",
        provider="anthropic",
        display_name="Claude Opus 4.6",
        context_window=200_000,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
    ),
    ModelInfo(
        id="claude-sonnet-4-5",
        provider="anthropic",
        display_name="Claude Sonnet 4.5",
        context_window=200_000,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
    ),
    ModelInfo(
        id="gpt-5.2",
        provider="openai",
        display_name="GPT-5.2",
        context_window=1_047_576,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
    ),
    ModelInfo(
        id="gpt-5.2-mini",
        provider="openai",
        display_name="GPT-5.2 Mini",
        context_window=1_047_576,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
    ),
    ModelInfo(
        id="gpt-5.2-codex",
        provider="openai",
        display_name="GPT-5.2 Codex",
        context_window=1_047_576,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
    ),
    ModelInfo(
        id="gemini-3-pro-preview",
        provider="gemini",
        display_name="Gemini 3 Pro Preview",
        context_window=1_048_576,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
    ),
    ModelInfo(
        id="gemini-3-flash-preview",
        provider="gemini",
        display_name="Gemini 3 Flash Preview",
        context_window=1_048_576,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
    ),
]

_CATALOG_INDEX: dict[str, ModelInfo] = {m.id: m for m in MODEL_CATALOG}

_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-5.2",
    "gemini": "gemini-3-flash-preview",
}


def get_model_info(model_id: str) -> ModelInfo | None:
    """Look up model metadata by ID.

    Returns:
        ModelInfo if found, None otherwise.
    """
    return _CATALOG_INDEX.get(model_id)


def list_models(provider: str | None = None) -> list[ModelInfo]:
    """List all known models, optionally filtered by provider."""
    if provider is None:
        return list(MODEL_CATALOG)
    return [m for m in MODEL_CATALOG if m.provider == provider]


def get_default_model(provider: str) -> ModelInfo:
    """Get the default model for a provider.

    Raises:
        KeyError: If provider is unknown.
    """
    model_id = _DEFAULT_MODELS.get(provider)
    if model_id is None:
        raise KeyError(f"Unknown provider: {provider!r}")
    info = _CATALOG_INDEX.get(model_id)
    if info is None:
        raise KeyError(f"Default model {model_id!r} not found in catalog")
    return info
