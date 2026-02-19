"""Live API tests covering coverage gaps across Waves 9-13.

P2: Live audio/document adapter test (2 tests)
P3: Auth error session lifecycle live (1 test)
P5: Live stream event flow (1 test)

All tests skip automatically when the required API key is absent.
"""

from __future__ import annotations

import os

import pytest

from attractor_agent.session import Session, SessionConfig, SessionState
from attractor_llm import (
    Client,
    ContentPart,
    DocumentData,
    Message,
    ProviderConfig,
    RetryPolicy,
    StreamEventKind,
    stream,
)
from attractor_llm.adapters.anthropic import AnthropicAdapter
from attractor_llm.adapters.gemini import GeminiAdapter

# ------------------------------------------------------------------ #
# Env-var helpers (mirrors test_live_wave9_10_p1.py conventions)
# ------------------------------------------------------------------ #

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_KEY = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")

HAS_OPENAI = bool(OPENAI_KEY)
HAS_ANTHROPIC = bool(ANTHROPIC_KEY)
HAS_GEMINI = bool(GEMINI_KEY)

skip_no_anthropic = pytest.mark.skipif(not HAS_ANTHROPIC, reason="ANTHROPIC_API_KEY not set")
skip_no_gemini = pytest.mark.skipif(not HAS_GEMINI, reason="GOOGLE_API_KEY not set")

ANTHROPIC_MODEL = "claude-sonnet-4-5"
GEMINI_MODEL = "gemini-2.0-flash"

# ------------------------------------------------------------------ #
# Minimal valid PDF bytes (used for document adapter tests)
# The structure is a well-formed 1-page PDF with no content -- small
# enough to be accepted by provider APIs without burning tokens.
# ------------------------------------------------------------------ #

_MINIMAL_PDF_BYTES = (
    b"%PDF-1.4\n"
    b"1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n"
    b"2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n"
    b"3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]>>\nendobj\n"
    b"xref\n0 4\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000058 00000 n \n"
    b"0000000115 00000 n \n"
    b"trailer\n<</Size 4 /Root 1 0 R>>\n"
    b"startxref\n194\n%%EOF"
)

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def anthropic_client() -> Client:
    client = Client(retry_policy=RetryPolicy(max_retries=1))
    client.register_adapter(
        "anthropic", AnthropicAdapter(ProviderConfig(api_key=ANTHROPIC_KEY, timeout=60.0))
    )
    return client


@pytest.fixture
def gemini_client() -> Client:
    client = Client(retry_policy=RetryPolicy(max_retries=1))
    client.register_adapter(
        "gemini", GeminiAdapter(ProviderConfig(api_key=GEMINI_KEY, timeout=60.0))
    )
    return client


# ================================================================== #
# P2: Live document adapter tests
# ================================================================== #


class TestLiveDocumentAdapters:
    """Sending a ContentPart with kind=DOCUMENT to real provider APIs.

    These tests verify that the adapter correctly translates a DocumentData
    payload and that the provider accepts the format. The model's ability
    to understand an empty/minimal PDF is irrelevant -- we just need no
    adapter-level crash and a valid response back.
    """

    @skip_no_gemini
    @pytest.mark.asyncio
    async def test_gemini_accepts_document_part(self, gemini_client: Client) -> None:
        """Gemini adapter translates DOCUMENT ContentPart to a valid API request.

        Sends a minimal PDF alongside a text instruction. The provider should
        return a response (possibly noting it cannot read the content), not
        raise an adapter error.
        """
        doc = DocumentData(
            data=_MINIMAL_PDF_BYTES,
            media_type="application/pdf",
            file_name="test.pdf",
        )
        messages = [
            Message.user_parts(
                ContentPart.document_part(doc),
                ContentPart.text_part("This document is a test fixture. Reply with just: ok"),
            )
        ]

        from attractor_llm.generate import generate

        result = await generate(
            gemini_client,
            GEMINI_MODEL,
            messages=messages,
            provider="gemini",
        )

        # As long as we get a response without an exception the adapter
        # translation is correct. We don't assert on the response text
        # since the model may note it can't read the minimal PDF.
        assert result is not None, "Expected a GenerateResult, got None"
        assert len(result.steps) >= 1, "Expected at least one step in the result"

    @skip_no_anthropic
    @pytest.mark.asyncio
    async def test_anthropic_accepts_document_part(self, anthropic_client: Client) -> None:
        """Anthropic adapter translates DOCUMENT ContentPart to a valid API request.

        Anthropic supports PDF documents via base64-encoded source blocks.
        The adapter should convert DocumentData(data=bytes) into that format.
        """
        doc = DocumentData(
            data=_MINIMAL_PDF_BYTES,
            media_type="application/pdf",
            file_name="test.pdf",
        )
        messages = [
            Message.user_parts(
                ContentPart.document_part(doc),
                ContentPart.text_part("This document is a test fixture. Reply with just: ok"),
            )
        ]

        from attractor_llm.generate import generate

        result = await generate(
            anthropic_client,
            ANTHROPIC_MODEL,
            messages=messages,
            provider="anthropic",
        )

        assert result is not None, "Expected a GenerateResult, got None"
        assert len(result.steps) >= 1, "Expected at least one step in the result"


# ================================================================== #
# P3: Auth error session lifecycle live
# ================================================================== #


class TestAuthErrorSessionLifecycleLive:
    """An auth failure (401) must immediately close the session (not leave it IDLE).

    Spec §9.11: AuthenticationError / AccessDeniedError transition the
    session to SessionState.CLOSED so it cannot be reused.
    """

    @skip_no_anthropic
    @pytest.mark.asyncio
    async def test_invalid_api_key_closes_session(self) -> None:
        """Session transitions to CLOSED after an Anthropic auth error.

        We deliberately use a syntactically plausible but invalid API key
        so that the provider returns 401. No retries are configured so the
        auth error surfaces immediately.
        """
        # Build a client with a clearly invalid Anthropic key.
        # RetryPolicy(max_retries=0) prevents any retry attempt; even
        # without it, AuthenticationError.retryable == False so the retry
        # engine would skip retries regardless.
        bad_client = Client(retry_policy=RetryPolicy(max_retries=0))
        bad_client.register_adapter(
            "anthropic",
            AnthropicAdapter(
                ProviderConfig(api_key="sk-ant-invalid-key-for-testing-xxx", timeout=15.0)
            ),
        )

        config = SessionConfig(model=ANTHROPIC_MODEL, provider="anthropic")
        session = Session(client=bad_client, config=config)

        # The session should start IDLE.
        assert session.state == SessionState.IDLE

        # submit() catches AuthenticationError and transitions to CLOSED.
        result = await session.submit("Hello")

        # Spec §9.11: auth error → SessionState.CLOSED
        assert session.state == SessionState.CLOSED, (
            f"Expected session.state == CLOSED after auth error, got {session.state!r}. "
            f"Response was: {result!r}"
        )

        # The returned text should mention the auth error (not raise an exception).
        assert "Authentication Error" in result or "auth" in result.lower(), (
            f"Expected error message in result, got: {result!r}"
        )


# ================================================================== #
# P5: Live stream event flow
# ================================================================== #


class TestLiveStreamEventFlow:
    """Streaming from a real provider via iter_events() yields the expected event types.

    Verifies that:
    - At least one TEXT_DELTA event is emitted (streaming text arrives)
    - A FINISH event is emitted at the end (stream terminates cleanly)
    """

    @skip_no_anthropic
    @pytest.mark.asyncio
    async def test_anthropic_stream_contains_text_delta_and_finish_events(
        self, anthropic_client: Client
    ) -> None:
        """iter_events() on an Anthropic stream produces TEXT_DELTA and FINISH events."""
        sr = await stream(
            anthropic_client,
            ANTHROPIC_MODEL,
            "Reply with just the word: hello",
            provider="anthropic",
        )

        event_kinds: list[StreamEventKind] = []
        async for event in sr.iter_events():
            event_kinds.append(event.kind)

        text_delta_events = [k for k in event_kinds if k == StreamEventKind.TEXT_DELTA]
        finish_events = [k for k in event_kinds if k == StreamEventKind.FINISH]

        assert len(text_delta_events) >= 1, (
            f"Expected at least one TEXT_DELTA event; got event kinds: {event_kinds}"
        )
        assert len(finish_events) >= 1, (
            f"Expected at least one FINISH event; got event kinds: {event_kinds}"
        )

        # FINISH must appear after at least one TEXT_DELTA in the sequence.
        # (Some providers emit a trailing USAGE after FINISH, so we don't
        # assert FINISH is absolutely last -- just that TEXT_DELTA precedes it.)
        first_text_delta_idx = next(
            i for i, k in enumerate(event_kinds) if k == StreamEventKind.TEXT_DELTA
        )
        first_finish_idx = next(i for i, k in enumerate(event_kinds) if k == StreamEventKind.FINISH)
        assert first_finish_idx > first_text_delta_idx, (
            f"Expected FINISH to come after TEXT_DELTA. All event kinds: {event_kinds}"
        )
