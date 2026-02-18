from __future__ import annotations

from dataclasses import dataclass, field

_CAPABILITY_FIELDS: dict[str, str] = {
    "tools": "supports_tools",
    "vision": "supports_vision",
    "reasoning": "supports_reasoning",
}


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
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    aliases: list[str] = field(default_factory=list)


MODEL_CATALOG: list[ModelInfo] = [
    ModelInfo(
        id="claude-opus-4-6",
        provider="anthropic",
        display_name="Claude Opus 4.6",
        context_window=200_000,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        aliases=["opus", "claude-opus", "opus-4-6"],
    ),
    ModelInfo(
        id="claude-sonnet-4-5",
        provider="anthropic",
        display_name="Claude Sonnet 4.5",
        context_window=200_000,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        aliases=["sonnet", "claude-sonnet", "sonnet-4-5"],
    ),
    ModelInfo(
        id="gpt-5.2",
        provider="openai",
        display_name="GPT-5.2",
        context_window=1_047_576,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        aliases=["gpt-5", "5.2"],
    ),
    ModelInfo(
        id="gpt-5.2-mini",
        provider="openai",
        display_name="GPT-5.2 Mini",
        context_window=1_047_576,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        aliases=["gpt-mini", "5.2-mini"],
    ),
    ModelInfo(
        id="gpt-5.2-codex",
        provider="openai",
        display_name="GPT-5.2 Codex",
        context_window=1_047_576,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        aliases=["gpt-codex", "5.2-codex"],
    ),
    ModelInfo(
        id="gemini-3-pro-preview",
        provider="gemini",
        display_name="Gemini 3 Pro Preview",
        context_window=1_048_576,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        aliases=["gemini-pro", "3-pro"],
    ),
    ModelInfo(
        id="gemini-3-flash-preview",
        provider="gemini",
        display_name="Gemini 3 Flash Preview",
        context_window=1_048_576,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
        aliases=["gemini-flash", "3-flash", "flash"],
    ),
]

_CATALOG_INDEX: dict[str, ModelInfo] = {m.id: m for m in MODEL_CATALOG}

_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-5.2",
    "gemini": "gemini-3-flash-preview",
}


def get_model_info(model_id: str) -> ModelInfo | None:
    """Look up model metadata by ID or alias.

    Exact ID match takes precedence; falls back to alias search.

    Returns:
        ModelInfo if found, None otherwise.
    """
    # Exact ID match first (existing behaviour, O(1))
    info = _CATALOG_INDEX.get(model_id)
    if info is not None:
        return info

    # Alias search — return first entry whose aliases contain model_id
    for entry in MODEL_CATALOG:
        if model_id in entry.aliases:
            return entry

    return None


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


def get_latest_model(provider: str, capability: str | None = None) -> ModelInfo | None:
    """Return the newest/best model for a provider, optionally filtered by capability.

    Catalog order acts as rank: the first entry per provider is the latest/best.

    Args:
        provider:   Provider name, e.g. ``"anthropic"``, ``"openai"``, ``"gemini"``.
        capability: Optional capability filter — one of ``"tools"``, ``"vision"``,
                    or ``"reasoning"``.  An unknown capability string will always
                    return ``None``.

    Returns:
        The first matching :class:`ModelInfo`, or ``None`` if no entry satisfies
        both the provider and (optional) capability filter.
    """
    if capability is not None and capability not in _CAPABILITY_FIELDS:
        # Unknown capability — no model can satisfy it
        return None

    for entry in MODEL_CATALOG:
        if entry.provider != provider:
            continue
        if capability is not None:
            field_name = _CAPABILITY_FIELDS[capability]
            if not getattr(entry, field_name):
                continue
        return entry

    return None
