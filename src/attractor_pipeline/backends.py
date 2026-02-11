"""CodergenBackend implementations that bridge the pipeline to LLM providers.

This module contains the concrete backends that connect the Attractor
pipeline engine to the Coding Agent Loop and the Unified LLM Client.

AgentLoopBackend: Wraps a coding agent Session as a CodergenBackend.
    Pipeline node -> Agent Session -> LLM Client -> Provider API

DirectLLMBackend: Calls the LLM Client directly (no agent loop).
    Pipeline node -> LLM Client -> Provider API
    Simpler, no tools, good for simple prompt-response nodes.
"""

from __future__ import annotations

from typing import Any

from attractor_agent.abort import AbortSignal
from attractor_agent.session import Session, SessionConfig
from attractor_agent.tools.core import ALL_CORE_TOOLS
from attractor_llm.client import Client
from attractor_llm.types import Message, Request
from attractor_pipeline.engine.runner import HandlerResult, Outcome
from attractor_pipeline.graph import Node


class AgentLoopBackend:
    """Bridges the Coding Agent Loop to the pipeline's CodergenBackend interface.

    Wraps a Client in an agent Session per call, giving the LLM access to
    developer tools (read_file, write_file, edit_file, shell, grep, glob).

    Usage::

        client = Client()
        client.register_adapter("anthropic", AnthropicAdapter(config))
        backend = AgentLoopBackend(client)

        # Register with pipeline handlers:
        registry.register("codergen", CodergenHandler(backend=backend))
    """

    def __init__(
        self,
        client: Client,
        *,
        default_model: str = "claude-sonnet-4-5",
        default_provider: str | None = None,
        system_prompt: str = "",
        include_tools: bool = True,
    ) -> None:
        self._client = client
        self._default_model = default_model
        self._default_provider = default_provider
        self._system_prompt = system_prompt
        self._include_tools = include_tools

    async def run(
        self,
        node: Node,
        prompt: str,
        context: dict[str, Any],
        abort_signal: AbortSignal | None = None,
    ) -> str | HandlerResult:
        """Execute an LLM call through the agent loop.

        Creates a fresh Session for each node execution, configured
        with the node's LLM settings (model, provider, reasoning_effort).
        """
        # Resolve model from node attrs or defaults
        model = node.llm_model or self._default_model
        provider = node.llm_provider or self._default_provider

        # Build session config from node attributes
        config = SessionConfig(
            model=model,
            provider=provider,
            system_prompt=self._system_prompt or f"You are working on: {context.get('goal', '')}",
            max_turns=1,  # Single-turn for pipeline nodes
            max_tool_rounds_per_turn=15,
            reasoning_effort=node.reasoning_effort or None,
        )

        # Create tools list
        tools = list(ALL_CORE_TOOLS) if self._include_tools else []

        # Run session with error handling
        try:
            async with Session(
                client=self._client,
                config=config,
                tools=tools,
                abort_signal=abort_signal,
            ) as session:
                result = await session.submit(prompt)
        except Exception as exc:  # noqa: BLE001
            return HandlerResult(
                status=Outcome.FAIL,
                failure_reason=f"{type(exc).__name__}: {exc}",
            )

        # Check for session-level error responses
        if result.startswith("[Error:") or result.startswith("[Session aborted]"):
            return HandlerResult(
                status=Outcome.FAIL,
                failure_reason=result,
                output=result,
            )

        return result


class DirectLLMBackend:
    """Calls the LLM Client directly without the agent loop.

    Simpler than AgentLoopBackend -- no tools, no multi-turn, just a
    single prompt-response call. Good for simple LLM tasks like
    summarization, classification, or text generation.
    """

    def __init__(
        self,
        client: Client,
        *,
        default_model: str = "claude-sonnet-4-5",
        default_provider: str | None = None,
    ) -> None:
        self._client = client
        self._default_model = default_model
        self._default_provider = default_provider

    async def run(
        self,
        node: Node,
        prompt: str,
        context: dict[str, Any],
        abort_signal: AbortSignal | None = None,
    ) -> str | HandlerResult:
        """Execute a single LLM call (no tools, no agent loop)."""
        model = node.llm_model or self._default_model
        provider = node.llm_provider or self._default_provider

        request = Request(
            model=model,
            provider=provider,
            messages=[Message.user(prompt)],
            system=f"You are working on: {context.get('goal', '')}",
            reasoning_effort=node.reasoning_effort or None,
        )

        try:
            response = await self._client.complete(request)
        except Exception as exc:  # noqa: BLE001
            return HandlerResult(
                status=Outcome.FAIL,
                failure_reason=f"{type(exc).__name__}: {exc}",
            )

        text = response.text or ""
        if not text:
            return HandlerResult(
                status=Outcome.FAIL,
                failure_reason="Empty response from LLM",
            )

        return text
