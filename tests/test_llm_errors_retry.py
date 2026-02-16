"""Tests for LLM SDK error hierarchy, classification, and retry engine."""

from __future__ import annotations

import pytest

from attractor_llm.errors import (
    AuthenticationError,
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
    classify_http_error,
)
from attractor_llm.retry import RetryPolicy

# ================================================================== #
# Error hierarchy
# ================================================================== #


class TestErrorHierarchy:
    def test_all_errors_inherit_from_sdk_error(self):
        errors = [
            ProviderError("test"),
            AuthenticationError("test"),
            RateLimitError("test"),
            ServerError("test"),
            ContentFilterError("test"),
            InvalidRequestError("test"),
            RequestTimeoutError("test"),
            StreamError("test"),
            ToolError("test"),
            SchemaValidationError("test"),
        ]
        for err in errors:
            assert isinstance(err, SDKError)

    def test_provider_errors_inherit_from_provider_error(self):
        assert isinstance(AuthenticationError("x"), ProviderError)
        assert isinstance(RateLimitError("x"), ProviderError)
        assert isinstance(ServerError("x"), ProviderError)
        assert isinstance(ContentFilterError("x"), ProviderError)


class TestRetryability:
    @pytest.mark.parametrize(
        ("error_cls", "expected"),
        [
            (AuthenticationError, False),
            (ContentFilterError, False),
            (InvalidRequestError, False),
            (SchemaValidationError, False),
            (ToolError, False),
            (RateLimitError, True),
            (ServerError, True),
            (RequestTimeoutError, True),
            (StreamError, True),
        ],
    )
    def test_retryability(self, error_cls, expected):
        err = error_cls("test")
        assert err.retryable is expected

    def test_rate_limit_retry_after(self):
        err = RateLimitError("slow down", retry_after=30.0)
        assert err.retry_after == 30.0
        assert err.retryable is True

    def test_tool_error_has_tool_name(self):
        err = ToolError("failed", tool_name="grep")
        assert err.tool_name == "grep"


# ================================================================== #
# classify_http_error
# ================================================================== #


class TestClassifyHttpError:
    def test_401_is_auth_error(self):
        err = classify_http_error(401, "unauthorized", "anthropic")
        assert isinstance(err, AuthenticationError)
        assert err.retryable is False
        assert err.provider == "anthropic"

    def test_403_is_auth_error(self):
        err = classify_http_error(403, "forbidden", "openai")
        assert isinstance(err, AuthenticationError)

    def test_429_is_rate_limit(self):
        err = classify_http_error(429, "too many", "anthropic")
        assert isinstance(err, RateLimitError)
        assert err.retryable is True

    def test_429_extracts_retry_after(self):
        err = classify_http_error(
            429,
            "too many",
            "openai",
            headers={"Retry-After": "30"},
        )
        assert isinstance(err, RateLimitError)
        assert err.retry_after == 30.0

    def test_429_retry_after_case_insensitive(self):
        err = classify_http_error(429, "x", "openai", headers={"retry-after": "15"})
        assert isinstance(err, RateLimitError)
        assert err.retry_after == 15.0

    def test_500_is_server_error(self):
        err = classify_http_error(500, "internal", "gemini")
        assert isinstance(err, ServerError)
        assert err.retryable is True

    def test_502_is_server_error(self):
        err = classify_http_error(502, "bad gateway", "anthropic")
        assert isinstance(err, ServerError)

    def test_408_is_retryable(self):
        err = classify_http_error(408, "timeout", "openai")
        assert isinstance(err, ProviderError)
        assert err.retryable is True

    def test_400_is_not_retryable(self):
        err = classify_http_error(400, "bad request", "anthropic")
        assert isinstance(err, ProviderError)
        assert err.retryable is False


# ================================================================== #
# RetryPolicy
# ================================================================== #


class TestRetryPolicy:
    def test_defaults(self):
        p = RetryPolicy()
        assert p.max_retries == 3
        assert p.initial_delay == 0.2
        assert p.backoff_factor == 2.0

    def test_compute_delay_exponential(self):
        p = RetryPolicy(initial_delay=1.0, backoff_factor=2.0, jitter=False)
        assert p.compute_delay(0) == 1.0
        assert p.compute_delay(1) == 2.0
        assert p.compute_delay(2) == 4.0

    def test_compute_delay_respects_max(self):
        p = RetryPolicy(initial_delay=1.0, backoff_factor=10.0, max_delay=5.0, jitter=False)
        assert p.compute_delay(0) == 1.0
        assert p.compute_delay(1) == 5.0  # capped at max_delay
        assert p.compute_delay(2) == 5.0

    def test_compute_delay_with_jitter(self):
        p = RetryPolicy(initial_delay=2.0, jitter=True)
        # With jitter, delay should be in [1.0, 2.0] (equal jitter)
        for _ in range(20):
            d = p.compute_delay(0)
            assert 1.0 <= d <= 2.0
