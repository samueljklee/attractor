"""Attractor Unified LLM Client SDK.

Provides a single interface across OpenAI, Anthropic, and Google Gemini.
"""

from __future__ import annotations

from attractor_llm.adapters.base import ProviderAdapter, ProviderConfig
from attractor_llm.catalog import ModelInfo, get_default_model, get_model_info, list_models
from attractor_llm.client import Client, get_default_client, set_default_client
from attractor_llm.errors import (
    AuthenticationError,
    ConfigurationError,
    ContentFilterError,
    InvalidRequestError,
    ProviderError,
    RateLimitError,
    RequestTimeoutError,
    SchemaValidationError,
    SDKError,
    ServerError,
    StreamError,
    ToolError,
)
from attractor_llm.retry import RetryPolicy, retry_with_policy
from attractor_llm.streaming import StreamAccumulator
from attractor_llm.types import (
    ContentPart,
    ContentPartKind,
    FinishReason,
    GenerateResult,
    ImageData,
    Message,
    Request,
    Response,
    Role,
    StepResult,
    StreamEvent,
    StreamEventKind,
    Tool,
    Usage,
)

__all__ = [
    # Client
    "Client",
    "get_default_client",
    "set_default_client",
    "ProviderAdapter",
    "ProviderConfig",
    # Types
    "Role",
    "ContentPartKind",
    "ContentPart",
    "ImageData",
    "Message",
    "Tool",
    "FinishReason",
    "Usage",
    "Request",
    "Response",
    "StepResult",
    "GenerateResult",
    "StreamEventKind",
    "StreamEvent",
    # Errors
    "SDKError",
    "ProviderError",
    "AuthenticationError",
    "RateLimitError",
    "ServerError",
    "ContentFilterError",
    "ConfigurationError",
    "InvalidRequestError",
    "RequestTimeoutError",
    "ToolError",
    "SchemaValidationError",
    "StreamError",
    # Retry
    "RetryPolicy",
    "retry_with_policy",
    # Catalog
    "ModelInfo",
    "get_model_info",
    "list_models",
    "get_default_model",
    # Streaming
    "StreamAccumulator",
]
