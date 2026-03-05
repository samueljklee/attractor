# Attractor Project -- Agent Instructions

## Swarm Review Configuration

When reviewing PRs, implementations, or plans, use this swarm of models across all 3 providers.
These are the **verified working model IDs** from `amplifier provider models <provider>`.

### Anthropic Provider

| Model ID | Display Name | Context | Best For |
|----------|-------------|---------|----------|
| `claude-opus-4-6` | Claude Opus 4.6 | 200K | Deep analysis, architecture review, security review |
| `claude-sonnet-4-6` | Claude Sonnet 4.6 | 200K | Implementation, code review, spec compliance |
| `claude-haiku-4-5-20251001` | Claude Haiku 4.5 | 200K | Fast reviews, simple tasks |

### OpenAI Provider

| Model ID | Display Name | Context | Best For |
|----------|-------------|---------|----------|
| `gpt-5.2` | GPT-5.2 | 400K | General review, large context analysis |
| `gpt-5.2-pro` | GPT-5.2 Pro | 400K | Deeper analysis, precision responses |
| `gpt-5.2-codex` | GPT-5.2 Codex | 400K | Code-focused review, implementation |
| `gpt-5.1` | GPT-5.1 | 400K | General review |
| `gpt-5` | GPT-5 | 400K | General review |
| `gpt-5-mini` | GPT-5 Mini | 400K | Fast, cost-efficient reviews |
| `gpt-5-pro` | GPT-5 Pro | 400K | Deep reasoning |

### Google Gemini Provider

| Model ID | Display Name | Context | Best For |
|----------|-------------|---------|----------|
| `gemini-3-pro-preview` | Gemini 3 Pro Preview | 1M | Large codebase analysis, completeness checks |
| `gemini-3-flash-preview` | Gemini 3 Flash Preview | 1M | Fast reviews |
| `gemini-2.5-pro` | Gemini 2.5 Pro | 1M | Deep analysis, thinking |
| `gemini-2.5-flash` | Gemini 2.5 Flash | 1M | Fast, cost-efficient |

### Default Swarm Configuration (use for all PR reviews)

Dispatch these 5 models in parallel for every review:

```
delegate(provider_preferences=[{provider: "anthropic", model: "claude-opus-4-6"}])
delegate(provider_preferences=[{provider: "anthropic", model: "claude-sonnet-4-6"}])
delegate(provider_preferences=[{provider: "anthropic", model: "claude-opus-4-5"}])  # falls back to opus-4-6
delegate(provider_preferences=[{provider: "openai", model: "gpt-5.2-pro"}])
delegate(provider_preferences=[{provider: "google", model: "gemini-3-pro-preview"}])
```

Note: `claude-opus-4-5` is not a separate model in the Amplifier catalog -- it routes to the
latest Opus. For true diversity, use different providers rather than different Anthropic versions.

### Recommended 8-Model Swarm (when maximum diversity needed)

```
# Anthropic (3 models)
delegate(provider_preferences=[{provider: "anthropic", model: "claude-opus-4-6"}])
delegate(provider_preferences=[{provider: "anthropic", model: "claude-sonnet-4-6"}])
delegate(provider_preferences=[{provider: "anthropic", model: "claude-haiku-4-5-20251001"}])

# OpenAI (3 models)
delegate(provider_preferences=[{provider: "openai", model: "gpt-5.2-pro"}])
delegate(provider_preferences=[{provider: "openai", model: "gpt-5.2-codex"}])
delegate(provider_preferences=[{provider: "openai", model: "gpt-5-mini"}])

# Google (2 models)
delegate(provider_preferences=[{provider: "google", model: "gemini-3-pro-preview"}])
delegate(provider_preferences=[{provider: "google", model: "gemini-2.5-pro"}])
```

## Workflow for This Project

### Standard PR Cycle

1. **Implement** on a feature branch
2. **Mock tests** -- run full suite, verify all pass
3. **Live tests** -- run e2e + comprehensive + 3-provider smoke against real APIs
4. **5-model swarm review** -- dispatch all 5 default models
5. **Fix** all findings from the swarm
6. **Re-test** after fixes
7. **Merge** to main

### Test Commands

```bash
# Mock tests (excludes live)
uv run python -m pytest tests/ --ignore=tests/test_e2e_integration.py --ignore=tests/test_live_comprehensive.py --ignore=tests/test_live_wave9_10_p1.py -x -q

# Live e2e
uv run python -m pytest tests/test_e2e_integration.py -v -x

# Live comprehensive (all 3 providers)
uv run python -m pytest tests/test_live_comprehensive.py -v -x

# Live wave9/10/p1
uv run python -m pytest tests/test_live_wave9_10_p1.py -v -x

# All tests
uv run python -m pytest tests/ -v -x
```

## Spec Compliance

This project implements the strongdm/attractor NL specs:
- `unified-llm-spec.md` -- LLM SDK (§8 DoD)
- `coding-agent-loop-spec.md` -- Agent Loop (§9 DoD)
- `attractor-spec.md` -- Pipeline Engine (§11 DoD)

Specs are fetched from: `https://raw.githubusercontent.com/strongdm/attractor/main/`

### Audit Recipe

Run the spec compliance audit when specs change:
```bash
amplifier recipe execute recipes/spec-compliance-audit.yaml
```

### Wave Implementation Recipe

Execute a wave from the approved plan:
```bash
amplifier recipe execute recipes/wave-implementation.yaml \
  --context '{"wave_name": "wave14-pipeline-graph-validation", "wave_items_description": "..."}'
```

### Current Progress

Waves 9-13 complete (28 of 58 items). Remaining: Waves 14-16b (30 items).
Wave plan at: `docs/plans/2026-02-17-attractor-spec-compliance-waves.json`

## Known Spec Extensions and API Gaps

These are intentional deviations from the spec, either extending it or
documenting places where the live API does not match the spec. Any AI
assistant working on this project must not silently revert these without
understanding the rationale.

### Spec Extensions (we added, spec does not define)

| What | Where | Why | Spec ref |
|------|-------|-----|----------|
| `knowledge_cutoff: str \| None` field on `ModelInfo` | `catalog.py` | coding-agent-loop-spec §6.3 requires `Knowledge cutoff: {date}` in the `<environment>` block. No provider API exposes this at runtime, so we hardcode it per model in the catalog. The unified-llm-spec §2.9 `ModelInfo` schema does not include this field — pending spec amendment. | §6.3 agent loop |
| `gpt-4.1-mini` catalog entry | `catalog.py` | Real working OpenAI model used in tests and examples. Not in spec §2.9 but confirmed live. | — |
| `gemini-2.5-pro`, `gemini-2.5-flash` catalog entries | `catalog.py` | Real working Gemini models with confirmed `generationConfig.thinkingConfig` support. Not in spec §2.9 but confirmed live. | — |

### API vs Spec Gaps (spec says X, live API says Y)

| What | Spec says | API says | Status |
|------|-----------|----------|--------|
| `gpt-5.2-mini` | Exists, `supports_reasoning=true` | Returns 404 — model not found | Kept in catalog per spec; use `gpt-5.2` for live reasoning calls |
| `gemini-3-pro-preview`, `gemini-3-flash-preview` reasoning | `supports_reasoning=true` | v1beta API rejects `thinkingConfig` for these model IDs | Catalog follows spec; use `gemini-2.5-pro/flash` for live reasoning calls |
| Gemini `thinkingConfig` placement | §3.9 says `thinkingConfig` param | Must be `generationConfig.thinkingConfig` in v1beta API | Adapter uses `generationConfig.thinkingConfig` (correct API form) |
