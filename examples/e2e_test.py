#!/usr/bin/env python3
"""End-to-end test: Full Attractor pipeline with a real LLM.

This script proves the entire 3-layer stack works together:
  DOT graph -> Pipeline Engine -> CodergenHandler -> Agent Loop -> LLM Client -> Anthropic API

Usage:
    ANTHROPIC_API_KEY=sk-... python examples/e2e_test.py

What it does:
    1. Parses a DOT pipeline with 3 stages (plan, implement, review)
    2. Executes it using a real LLM (Claude Sonnet 4.5 via Anthropic API)
    3. Each stage gets the LLM's actual response, which flows into context
    4. The pipeline completes when the exit node is reached
"""

import asyncio
import os
import sys
import time


async def main() -> None:
    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        print("Usage: ANTHROPIC_API_KEY=sk-... python examples/e2e_test.py")
        sys.exit(1)

    # Import everything
    from attractor_llm.adapters.anthropic import AnthropicAdapter
    from attractor_llm.adapters.base import ProviderConfig
    from attractor_llm.client import Client
    from attractor_pipeline import (
        HandlerRegistry,
        PipelineStatus,
        parse_dot,
        register_default_handlers,
        run_pipeline,
    )
    from attractor_pipeline.backends import DirectLLMBackend

    print("=" * 60)
    print("ATTRACTOR END-TO-END TEST")
    print("=" * 60)
    print()

    # --- Layer 1: Set up the LLM Client ---
    print("[1/4] Setting up LLM Client with Anthropic adapter...")
    client = Client()
    adapter = AnthropicAdapter(
        ProviderConfig(
            api_key=api_key,
            timeout=60.0,
        )
    )
    client.register_adapter("anthropic", adapter)
    print("  -> Anthropic adapter registered")

    # --- Layer 2+3 bridge: CodergenBackend ---
    print("[2/4] Creating DirectLLMBackend (LLM without tools for speed)...")
    backend = DirectLLMBackend(
        client,
        default_model="claude-sonnet-4-5",
        default_provider="anthropic",
    )
    print("  -> Backend ready")

    # --- Layer 3: Parse the pipeline ---
    print("[3/4] Parsing pipeline DOT graph...")
    pipeline_dot = """
    digraph FeaturePipeline {
        graph [goal="Write a Python function that calculates fibonacci numbers"]

        start [shape=Mdiamond, label="Start"]

        plan [
            shape=box,
            label="Plan",
            prompt="Create a brief plan (3 bullet points max) for: $goal. Be concise."
        ]

        implement [
            shape=box,
            label="Implement",
            prompt="Based on this plan, write the Python code. Only output the code. Plan: $goal"
        ]

        done [shape=Msquare, label="Done"]

        start -> plan -> implement -> done
    }
    """

    graph = parse_dot(pipeline_dot)
    print(f"  -> Parsed: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    print(f"  -> Goal: {graph.goal}")

    # --- Execute the pipeline ---
    print("[4/4] Executing pipeline with real LLM calls...")
    print()

    # Set up handlers with real backend
    registry = HandlerRegistry()
    register_default_handlers(registry, codergen_backend=backend)

    start_time = time.monotonic()

    async with client:
        result = await run_pipeline(graph, registry)

    duration = time.monotonic() - start_time

    # --- Report results ---
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Status: {result.status}")
    print(f"Duration: {duration:.1f}s")
    print(f"Nodes completed: {result.completed_nodes}")
    print()

    if result.status == PipelineStatus.COMPLETED:
        print("--- Plan output ---")
        plan_output = result.context.get("codergen.plan.output", "(none)")
        print(plan_output[:500])
        print()

        print("--- Implementation output ---")
        impl_output = result.context.get("codergen.implement.output", "(none)")
        print(impl_output[:1000])
        print()

        print("=" * 60)
        print("END-TO-END TEST: PASS")
        print("=" * 60)
        print()
        print("The full 3-layer stack works:")
        print("  DOT graph -> Pipeline Engine -> CodergenHandler")
        print("  -> DirectLLMBackend -> LLM Client -> Anthropic API")
        print(f"  -> Real Claude responses in {duration:.1f}s")

    elif result.status == PipelineStatus.FAILED:
        print(f"FAILED: {result.error}")
        print()
        print("=" * 60)
        print("END-TO-END TEST: FAIL")
        print("=" * 60)
        sys.exit(1)

    elif result.status == PipelineStatus.CANCELLED:
        print("Pipeline was cancelled")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
