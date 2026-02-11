"""CLI entry point for Attractor pipeline runner.

Usage:
    attractor run pipeline.dot                    # Run with Anthropic (default)
    attractor run pipeline.dot --provider openai  # Run with OpenAI
    attractor run pipeline.dot --model gpt-5.2    # Specify model
    attractor run pipeline.dot --validate-only    # Just validate, don't execute
    attractor validate pipeline.dot               # Validate a DOT file
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

from attractor_pipeline.validation import Severity, validate


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="attractor",
        description="DOT-based pipeline runner for multi-stage AI workflows",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- run command ---
    run_parser = subparsers.add_parser("run", help="Execute a DOT pipeline")
    run_parser.add_argument("dotfile", type=str, help="Path to the DOT pipeline file")
    run_parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="LLM provider (anthropic, openai, gemini). Auto-detected from model.",
    )
    run_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model ID. Default: claude-sonnet-4-5",
    )
    run_parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the DOT file without executing",
    )
    run_parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Use DirectLLMBackend (no agent tools)",
    )
    run_parser.add_argument(
        "--logs-dir",
        type=str,
        default=None,
        help="Directory for logs and checkpoints",
    )

    # --- validate command ---
    val_parser = subparsers.add_parser("validate", help="Validate a DOT pipeline file")
    val_parser.add_argument("dotfile", type=str, help="Path to the DOT pipeline file")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "validate":
        _cmd_validate(args.dotfile)
    elif args.command == "run":
        if args.validate_only:
            _cmd_validate(args.dotfile)
        else:
            asyncio.run(_cmd_run(args))


def _cmd_validate(dotfile: str) -> None:
    """Validate a DOT file and print diagnostics."""
    from attractor_pipeline.parser import parse_dot
    from attractor_pipeline.parser.parser import ParseError

    path = Path(dotfile)
    if not path.exists():
        print(f"Error: File not found: {dotfile}")
        sys.exit(1)

    source = path.read_text(encoding="utf-8")

    # Parse
    try:
        graph = parse_dot(source)
    except ParseError as e:
        print(f"Parse error: {e}")
        sys.exit(1)

    print(f"Parsed: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    if graph.goal:
        print(f"Goal: {graph.goal}")

    # Validate
    diagnostics = validate(graph)

    if not diagnostics:
        print("Validation: PASS (no issues)")
        return

    errors = 0
    warnings = 0
    for d in diagnostics:
        icon = {"error": "E", "warning": "W", "info": "I"}[d.severity]
        loc = f" (node: {d.node_id})" if d.node_id else ""
        print(f"  [{icon}] {d.rule}: {d.message}{loc}")
        if d.severity == Severity.ERROR:
            errors += 1
        elif d.severity == Severity.WARNING:
            warnings += 1

    print(
        f"\nValidation: {errors} error(s), {warnings} warning(s), "
        f"{len(diagnostics) - errors - warnings} info"
    )

    if errors > 0:
        print("FAIL: Fix errors before running this pipeline.")
        sys.exit(1)


async def _cmd_run(args: argparse.Namespace) -> None:
    """Execute a DOT pipeline."""
    from attractor_llm.client import Client
    from attractor_pipeline import (
        HandlerRegistry,
        PipelineStatus,
        parse_dot,
        register_default_handlers,
        run_pipeline,
    )
    from attractor_pipeline.backends import AgentLoopBackend, DirectLLMBackend
    from attractor_pipeline.parser.parser import ParseError
    from attractor_pipeline.validation import validate_or_raise

    dotfile = args.dotfile
    path = Path(dotfile)
    if not path.exists():
        print(f"Error: File not found: {dotfile}")
        sys.exit(1)

    source = path.read_text(encoding="utf-8")

    # Parse
    try:
        graph = parse_dot(source)
    except ParseError as e:
        print(f"Parse error: {e}")
        sys.exit(1)

    # Validate
    try:
        validate_or_raise(graph)
    except ValueError as e:
        print(str(e))
        sys.exit(1)

    print(f"Pipeline: {graph.name}")
    print(f"Goal: {graph.goal or '(none)'}")
    print(f"Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}")
    print()

    # Resolve provider and model
    model = args.model or "claude-sonnet-4-5"
    provider = args.provider

    # Auto-detect provider from model name
    if provider is None:
        if model.startswith("claude"):
            provider = "anthropic"
        elif model.startswith(("gpt", "o1", "o3", "o4")):
            provider = "openai"
        elif model.startswith("gemini"):
            provider = "gemini"
        else:
            print(
                f"Warning: Could not auto-detect provider from model '{model}'. "
                f"Defaulting to anthropic. Use --provider to specify explicitly."
            )
            provider = "anthropic"

    # Get API key
    key_env_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GOOGLE_API_KEY",
    }
    env_var = key_env_map.get(provider, "ANTHROPIC_API_KEY")
    api_key = os.environ.get(env_var)
    if not api_key:
        print(f"Error: Set {env_var} environment variable")
        sys.exit(1)

    # Set up LLM client
    client = Client()
    adapter = _create_adapter(provider, api_key)
    client.register_adapter(provider, adapter)
    print(f"Provider: {provider} ({model})")

    # Set up backend
    if args.no_tools:
        backend = DirectLLMBackend(
            client,
            default_model=model,
            default_provider=provider,
        )
        print("Backend: DirectLLM (no tools)")
    else:
        backend = AgentLoopBackend(
            client,
            default_model=model,
            default_provider=provider,
        )
        print("Backend: AgentLoop (with tools)")

    # Set up handlers
    registry = HandlerRegistry()
    register_default_handlers(registry, codergen_backend=backend)

    # Set up logs directory
    logs_root = None
    if args.logs_dir:
        logs_root = Path(args.logs_dir)
        logs_root.mkdir(parents=True, exist_ok=True)

    # Execute
    print()
    print("Executing pipeline...")
    print("-" * 40)
    start_time = time.monotonic()

    async with client:
        result = await run_pipeline(
            graph,
            registry,
            logs_root=logs_root,
        )

    duration = time.monotonic() - start_time

    # Report results
    print("-" * 40)
    print()
    print(f"Status: {result.status}")
    print(f"Duration: {duration:.1f}s")
    print(f"Nodes: {' -> '.join(result.completed_nodes)}")

    if result.error:
        print(f"Error: {result.error}")

    # Print codergen outputs
    print()
    for key, value in result.context.items():
        if key.startswith("codergen.") and key.endswith(".output"):
            node_id = key.split(".")[1]
            print(f"--- {node_id} output ---")
            print(str(value)[:2000])
            print()

    if result.status == PipelineStatus.COMPLETED:
        print("Pipeline completed successfully.")
    else:
        sys.exit(1)


def _create_adapter(provider: str, api_key: str) -> Any:
    """Create the appropriate provider adapter."""
    from attractor_llm.adapters.base import ProviderConfig

    config = ProviderConfig(api_key=api_key, timeout=120.0)

    if provider == "anthropic":
        from attractor_llm.adapters.anthropic import AnthropicAdapter

        return AnthropicAdapter(config)
    elif provider == "openai":
        from attractor_llm.adapters.openai import OpenAIAdapter

        return OpenAIAdapter(config)
    elif provider == "gemini":
        from attractor_llm.adapters.gemini import GeminiAdapter

        return GeminiAdapter(config)
    else:
        print(f"Unknown provider: {provider}")
        sys.exit(1)


if __name__ == "__main__":
    main()
