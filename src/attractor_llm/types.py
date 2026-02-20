"""Core data model for the Unified LLM Client SDK.

Implements the data types from the Unified LLM Client Specification §3.
All types use Pydantic v2 for validation and serialization.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, Field, computed_field, model_validator


class Role(StrEnum):
    """Message roles. Spec §3.1."""

    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ContentPartKind(StrEnum):
    """Content part types. Spec §3.2."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    DOCUMENT = "document"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    REDACTED_THINKING = "redacted_thinking"


class ImageData(BaseModel):
    """Image content, either inline bytes or URL. Spec §3.2, §3.3."""

    data: bytes | None = None
    url: str | None = None
    file_path: str | None = None
    media_type: str = "image/png"

    @model_validator(mode="after")
    def _check_source(self) -> Self:
        if self.data is None and self.url is None and self.file_path is None:
            raise ValueError("ImageData requires either 'data', 'url', or 'file_path'")
        return self

    @classmethod
    def from_file(cls, path: str) -> ImageData:
        """Create ImageData by reading a local file. Spec §3.3.

        Reads the file contents into ``data`` and infers ``media_type``
        from the file extension.

        Args:
            path: Path to a local image file.

        Returns:
            ImageData with ``data`` and ``media_type`` populated.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not a recognized image type.
        """
        import mimetypes
        from pathlib import Path as _Path

        p = _Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        mime, _ = mimetypes.guess_type(str(p))
        if mime is None or not mime.startswith("image/"):
            mime = "application/octet-stream"

        raw = p.read_bytes()
        return cls(data=raw, file_path=str(p), media_type=mime)


class AudioData(BaseModel):
    """Audio content, either inline bytes or URL. Spec §3.5."""

    url: str | None = None
    data: bytes | None = None
    media_type: str | None = None

    @model_validator(mode="after")
    def _check_source(self) -> Self:
        if self.data is None and self.url is None:
            raise ValueError("AudioData requires either 'data' or 'url'")
        return self


class DocumentData(BaseModel):
    """Document content, either inline bytes or URL. Spec §3.5."""

    url: str | None = None
    data: bytes | None = None
    media_type: str | None = None
    file_name: str | None = None

    @model_validator(mode="after")
    def _check_source(self) -> Self:
        if self.data is None and self.url is None:
            raise ValueError("DocumentData requires either 'data' or 'url'")
        return self


class ContentPart(BaseModel):
    """Tagged union for message content parts. Spec §3.2.

    Each part has a `kind` discriminator and kind-specific fields.
    Only fields relevant to the kind should be set. A model validator
    enforces required fields per kind at construction time.
    """

    kind: ContentPartKind

    # TEXT / THINKING
    text: str | None = None

    # IMAGE
    image: ImageData | None = None

    # AUDIO (Spec §3.5)
    audio: AudioData | None = None

    # DOCUMENT (Spec §3.5)
    document: DocumentData | None = None

    # TOOL_CALL / TOOL_RESULT
    tool_call_id: str | None = None
    name: str | None = None
    arguments: str | dict[str, Any] | None = None  # TOOL_CALL only

    # TOOL_RESULT only
    output: str | None = None
    is_error: bool = False

    # THINKING only
    signature: str | None = None

    # REDACTED_THINKING only
    redacted_data: str | None = None

    @model_validator(mode="after")
    def _validate_kind_fields(self) -> Self:
        """Enforce that required fields are set for each kind."""
        match self.kind:
            case ContentPartKind.TEXT:
                if self.text is None:
                    raise ValueError("TEXT content part requires 'text'")
            case ContentPartKind.IMAGE:
                if self.image is None:
                    raise ValueError("IMAGE content part requires 'image'")
            case ContentPartKind.TOOL_CALL:
                if self.tool_call_id is None or self.name is None:
                    raise ValueError("TOOL_CALL content part requires 'tool_call_id' and 'name'")
            case ContentPartKind.TOOL_RESULT:
                if self.tool_call_id is None or self.name is None:
                    raise ValueError("TOOL_RESULT content part requires 'tool_call_id' and 'name'")
            case ContentPartKind.THINKING:
                if self.text is None:
                    raise ValueError("THINKING content part requires 'text'")
            case ContentPartKind.REDACTED_THINKING:
                if self.redacted_data is None:
                    raise ValueError("REDACTED_THINKING content part requires 'redacted_data'")
            case ContentPartKind.AUDIO:
                if self.audio is None:
                    raise ValueError("AUDIO content part requires 'audio'")
            case ContentPartKind.DOCUMENT:
                if self.document is None:
                    raise ValueError("DOCUMENT content part requires 'document'")
        return self

    @classmethod
    def text_part(cls, text: str) -> ContentPart:
        return cls(kind=ContentPartKind.TEXT, text=text)

    @classmethod
    def image_part(cls, image: ImageData) -> ContentPart:
        return cls(kind=ContentPartKind.IMAGE, image=image)

    @classmethod
    def tool_call_part(
        cls, tool_call_id: str, name: str, arguments: str | dict[str, Any]
    ) -> ContentPart:
        return cls(
            kind=ContentPartKind.TOOL_CALL,
            tool_call_id=tool_call_id,
            name=name,
            arguments=arguments,
        )

    @classmethod
    def tool_result_part(
        cls, tool_call_id: str, name: str, output: str, is_error: bool = False
    ) -> ContentPart:
        return cls(
            kind=ContentPartKind.TOOL_RESULT,
            tool_call_id=tool_call_id,
            name=name,
            output=output,
            is_error=is_error,
        )

    @classmethod
    def thinking_part(cls, text: str, signature: str | None = None) -> ContentPart:
        return cls(kind=ContentPartKind.THINKING, text=text, signature=signature)

    @classmethod
    def audio_part(cls, audio: AudioData) -> ContentPart:
        return cls(kind=ContentPartKind.AUDIO, audio=audio)

    @classmethod
    def document_part(cls, document: DocumentData) -> ContentPart:
        return cls(kind=ContentPartKind.DOCUMENT, document=document)

    @classmethod
    def redacted_thinking_part(cls, redacted_data: str) -> ContentPart:
        return cls(kind=ContentPartKind.REDACTED_THINKING, redacted_data=redacted_data)


class Message(BaseModel):
    """A conversation message with role and content parts. Spec §3.1."""

    role: Role
    content: list[ContentPart]

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role=Role.USER, content=[ContentPart.text_part(text)])

    @classmethod
    def user_parts(cls, *parts: ContentPart) -> Message:
        """Create a user message with multiple content parts (e.g. text + image)."""
        return cls(role=Role.USER, content=list(parts))

    @classmethod
    def assistant(cls, text: str) -> Message:
        return cls(role=Role.ASSISTANT, content=[ContentPart.text_part(text)])

    @classmethod
    def system(cls, text: str) -> Message:
        return cls(role=Role.SYSTEM, content=[ContentPart.text_part(text)])

    @classmethod
    def developer(cls, text: str) -> Message:
        return cls(role=Role.DEVELOPER, content=[ContentPart.text_part(text)])

    @classmethod
    def tool_result(
        cls,
        tool_call_id: str,
        name: str,
        output: str,
        is_error: bool = False,
    ) -> Message:
        return cls(
            role=Role.TOOL,
            content=[ContentPart.tool_result_part(tool_call_id, name, output, is_error)],
        )

    @property
    def text(self) -> str | None:
        """Convenience: first TEXT content part's text."""
        for part in self.content:
            if part.kind == ContentPartKind.TEXT and part.text is not None:
                return part.text
        return None

    @property
    def tool_calls(self) -> list[ContentPart]:
        """All TOOL_CALL content parts."""
        return [p for p in self.content if p.kind == ContentPartKind.TOOL_CALL]

    @property
    def thinking(self) -> list[ContentPart]:
        """All THINKING content parts."""
        return [p for p in self.content if p.kind == ContentPartKind.THINKING]


# Tool execute function type
ToolExecuteFunc = Callable[..., Coroutine[Any, Any, str]]


class Tool(BaseModel):
    """A tool the LLM can call. Spec §5.1."""

    model_config = {"arbitrary_types_allowed": True}

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    execute: ToolExecuteFunc | None = Field(default=None, exclude=True)


class FinishReason(StrEnum):
    """Why the model stopped generating. Spec §3.6."""

    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    MAX_TOKENS = "max_tokens"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


class Usage(BaseModel):
    """Token usage tracking with aggregation support. Spec §3.8.

    Note: total_tokens = input_tokens + output_tokens. Reasoning tokens
    are a subset of output_tokens (billed as output but not visible).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    """Approximate reasoning/thinking tokens. For Anthropic, estimated as
    len(thinking_text) // 4 since the API does not expose a native count.
    For OpenAI, sourced from output_tokens_details.reasoning_tokens (native).
    For Gemini, sourced from usageMetadata.thoughtsTokenCount when available.
    Reasoning tokens are a subset of output_tokens (billed as output but not
    visible in the response text).  Spec §8.9.29."""
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


class Request(BaseModel):
    """Unified LLM request. Spec §3.3."""

    model: str
    messages: list[Message] = Field(default_factory=list)
    system: str | None = None
    tools: list[Tool] | None = None
    tool_choice: str | None = None  # "auto" | "none" | "required" | tool name
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    seed: int | None = None
    reasoning_effort: str | None = None  # "low" | "medium" | "high"
    response_format: dict[str, Any] | None = None
    provider: str | None = None
    provider_options: dict[str, Any] | None = None

    def effective_messages(self) -> list[Message]:
        """Messages with system message prepended if set via ``system`` shortcut.

        Note: if both ``system`` and a SYSTEM message exist in ``messages``,
        the ``system`` shortcut is prepended first (resulting in two system messages).
        """
        msgs = list(self.messages)
        if self.system:
            msgs.insert(0, Message.system(self.system))
        return msgs

    @classmethod
    def simple(cls, model: str, prompt: str, **kwargs: Any) -> Request:
        """Shorthand for a single-turn text request."""
        return cls(model=model, messages=[Message.user(prompt)], **kwargs)


class Response(BaseModel):
    """Unified LLM response. Spec §3.4."""

    id: str = ""
    model: str = ""
    provider: str = ""
    message: Message = Field(default_factory=lambda: Message(role=Role.ASSISTANT, content=[]))
    finish_reason: FinishReason = FinishReason.STOP
    usage: Usage = Field(default_factory=Usage)
    raw: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)

    @property
    def text(self) -> str | None:
        """Convenience: first TEXT content from the response message."""
        return self.message.text

    @property
    def tool_calls(self) -> list[ContentPart]:
        """All tool calls in the response."""
        return self.message.tool_calls

    @property
    def reasoning(self) -> list[ContentPart]:
        """All thinking/reasoning content."""
        return self.message.thinking


class StreamEventKind(StrEnum):
    """Stream event types. Spec §3.13 / §3.14."""

    START = "start"
    TEXT_START = "text_start"
    TEXT_DELTA = "text_delta"
    TEXT_END = "text_end"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_END = "tool_call_end"
    THINKING_DELTA = "thinking_delta"
    REASONING_START = "reasoning_start"
    REASONING_END = "reasoning_end"
    USAGE = "usage"
    FINISH = "finish"
    ERROR = "error"
    PROVIDER_EVENT = "provider_event"
    # Spec §3.14 canonical names (preferred for new code; START/FINISH kept for compat)
    STREAM_START = "stream_start"
    STREAM_END = "stream_end"


class StreamEvent(BaseModel):
    """A single event from a streaming response. Spec §3.13."""

    kind: StreamEventKind

    # START metadata
    response_id: str | None = None
    model: str | None = None
    provider: str | None = None

    # TEXT_DELTA / THINKING_DELTA
    text: str | None = None

    # TOOL_CALL_START / TOOL_CALL_DELTA / TOOL_CALL_END
    tool_call_id: str | None = None
    tool_name: str | None = None
    arguments_delta: str | None = None

    # THINKING signature (carried on final THINKING_DELTA or FINISH)
    thinking_signature: str | None = None

    # USAGE
    usage: Usage | None = None

    # FINISH
    finish_reason: FinishReason | None = None

    # ERROR
    error: str | None = None

    # PROVIDER_EVENT
    raw_event: dict[str, Any] | None = None


# ------------------------------------------------------------------ #
# Generate result types (Spec §4.3)
# ------------------------------------------------------------------ #


@dataclass
class StepResult:
    """Result of a single LLM call step within generate(). Spec §4.3.

    Each step represents one round-trip to the LLM, potentially
    followed by tool executions.

    The convenience properties below delegate to ``self.response`` so
    callers can write ``step.text`` instead of ``step.response.text``.
    All properties are read-only and produce no extra storage.
    """

    response: Response
    tool_results: list[ContentPart] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # P10: Spec §4.3 convenience accessors — delegate to response
    # ------------------------------------------------------------------ #

    @property
    def text(self) -> str | None:
        """First TEXT content from the response message."""
        return self.response.text

    @property
    def reasoning(self) -> list[ContentPart]:
        """All THINKING/reasoning content parts from the response."""
        return self.response.reasoning

    @property
    def tool_calls(self) -> list[ContentPart]:
        """All TOOL_CALL content parts from the response."""
        return self.response.tool_calls

    @property
    def finish_reason(self) -> FinishReason:
        """Why the model stopped generating."""
        return self.response.finish_reason

    @property
    def usage(self) -> Usage:
        """Token usage for this step."""
        return self.response.usage

    @property
    def warnings(self) -> list[str]:
        """Warnings emitted during this step (from the provider response)."""
        return self.response.warnings


@dataclass
class GenerateResult:
    """Result of generate() with step tracking. Spec §4.3.

    Backward-compatible: str(result) returns the text, and
    result == "some string" compares against the text.
    """

    text: str = ""
    steps: list[StepResult] = field(default_factory=list)
    total_usage: Usage = field(default_factory=Usage)

    def __str__(self) -> str:
        return self.text

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.text == other
        if isinstance(other, GenerateResult):
            return self.text == other.text
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.text)

    def __contains__(self, item: str) -> bool:
        return item in self.text

    def __bool__(self) -> bool:
        return bool(self.text)


@dataclass
class GenerateObjectResult(GenerateResult):
    """Result of generate_object() with the parsed JSON object. Spec §8.4.7.

    Extends GenerateResult so callers have access to step history, usage,
    and the raw JSON text in addition to the parsed dict.

    Backward-compatible: ``result == some_dict`` compares against
    ``parsed_object``, and ``result["key"]`` / ``result.keys()`` /
    iteration all delegate to ``parsed_object`` so existing callers that
    treat the return value as a plain dict continue to work.

    Attributes:
        parsed_object: The parsed JSON dict returned by the model.
    """

    parsed_object: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Dict-compatibility shim (backward compat for callers expecting dict)
    # ------------------------------------------------------------------ #

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        if isinstance(other, dict):
            return self.parsed_object == other
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(self.text)

    def __getitem__(self, key: str) -> Any:
        return self.parsed_object[key]

    def __iter__(self) -> Any:  # type: ignore[override]
        return iter(self.parsed_object)

    def __len__(self) -> int:
        return len(self.parsed_object)

    def __contains__(self, item: object) -> bool:
        return item in self.parsed_object

    def keys(self) -> Any:
        return self.parsed_object.keys()

    def values(self) -> Any:
        return self.parsed_object.values()

    def items(self) -> Any:
        return self.parsed_object.items()

    def get(self, key: str, default: Any = None) -> Any:
        return self.parsed_object.get(key, default)


# ------------------------------------------------------------------ #
# Timeout configuration (Spec §4.7)
# ------------------------------------------------------------------ #


@dataclass
class TimeoutConfig:
    """High-level timeout configuration for generate() calls. Spec §4.7.

    Controls overall and per-step timeouts for the generate() tool loop.
    """

    total: float | None = None
    per_step: float | None = None


@dataclass
class AdapterTimeout:
    """Low-level HTTP timeout configuration for provider adapters. Spec §4.7.

    Fine-grained control over HTTP connection, request, and stream timeouts.
    Passed via ProviderConfig.adapter_timeout to configure httpx clients.
    """

    connect: float = 10.0
    request: float = 120.0
    stream_read: float = 30.0
