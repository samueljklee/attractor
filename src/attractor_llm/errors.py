"""Error hierarchy for the Unified LLM Client SDK.

Implements the error taxonomy from the Unified LLM Client Specification §6.
Each error type carries retryability information for the retry engine.
"""

from __future__ import annotations


class SDKError(Exception):
    """Base error for all SDK errors."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable


class ProviderError(SDKError):
    """Error from a provider API response."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status_code: int | None = None,
        retryable: bool = False,
        raw_response: dict | None = None,  # noqa: UP006
    ) -> None:
        super().__init__(message, provider=provider, status_code=status_code, retryable=retryable)
        self.raw_response = raw_response


class AuthenticationError(ProviderError):
    """401: Invalid or missing credentials. Not retryable."""


class RateLimitError(ProviderError):
    """429: Rate limited. Retryable with backoff."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status_code: int | None = None,
        retry_after: float | None = None,
        raw_response: dict | None = None,  # noqa: UP006
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            status_code=status_code,
            retryable=True,
            raw_response=raw_response,
        )
        self.retry_after = retry_after


class ServerError(ProviderError):
    """5xx: Provider server error. Retryable."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status_code: int | None = None,
        raw_response: dict | None = None,  # noqa: UP006
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            status_code=status_code,
            retryable=True,
            raw_response=raw_response,
        )


class ContentFilterError(ProviderError):
    """Content was blocked by safety filters. Not retryable."""


class AccessDeniedError(ProviderError):
    """403: Insufficient permissions. Not retryable."""


class NotFoundError(ProviderError):
    """404: Model or endpoint not found. Not retryable."""


class ContextLengthError(ProviderError):
    """413: Input + output exceeds context window. Not retryable."""


class QuotaExceededError(ProviderError):
    """Billing/usage quota exhausted. Not retryable."""


class InvalidRequestError(ProviderError):
    """Bad request parameters. Not retryable."""


class AbortError(SDKError):
    """Request cancelled via abort signal. Not retryable."""


class NetworkError(SDKError):
    """Network-level failure. Retryable."""

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        super().__init__(message, provider=provider, retryable=True)


class InvalidToolCallError(SDKError):
    """Tool call arguments failed validation. Not retryable."""

    def __init__(self, message: str, *, tool_name: str | None = None) -> None:
        super().__init__(message, retryable=False)
        self.tool_name = tool_name


class NoObjectGeneratedError(SDKError):
    """Structured output parsing/validation failed. Not retryable."""


class ConfigurationError(SDKError):
    """SDK misconfiguration. Not retryable. Spec §6.

    Raised when the SDK is not properly configured (e.g., no providers
    registered, missing API keys).
    """

    def __init__(
        self, message: str, *, provider: str | None = None, status_code: int | None = None
    ) -> None:
        super().__init__(message, provider=provider, status_code=status_code, retryable=False)


class RequestTimeoutError(SDKError):
    """Request timed out. Retryable."""

    def __init__(
        self, message: str, *, provider: str | None = None, status_code: int | None = None
    ) -> None:
        super().__init__(message, provider=provider, status_code=status_code, retryable=True)


class StreamError(SDKError):
    """Stream was interrupted. Retryable by default."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(message, provider=provider, retryable=retryable)


class ToolError(SDKError):
    """Tool execution failed."""

    def __init__(self, message: str, *, tool_name: str | None = None) -> None:
        super().__init__(message, retryable=False)
        self.tool_name = tool_name


class SchemaValidationError(SDKError):
    """Schema or response validation failed. Not retryable."""

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        super().__init__(message, provider=provider, retryable=False)


def classify_http_error(
    status_code: int,
    body: str,
    provider: str,
    *,
    headers: dict[str, str] | None = None,
    raw_response: dict | None = None,  # noqa: UP006
) -> SDKError:
    """Map an HTTP status code to the appropriate error type. Spec §6.4."""
    # --- Status-code based classification ---
    if status_code in (400, 422):
        return InvalidRequestError(
            body, provider=provider, status_code=status_code, raw_response=raw_response
        )
    if status_code == 401:
        return AuthenticationError(
            body, provider=provider, status_code=status_code, raw_response=raw_response
        )
    if status_code == 403:
        return AccessDeniedError(
            body, provider=provider, status_code=status_code, raw_response=raw_response
        )
    if status_code == 404:
        return NotFoundError(
            body, provider=provider, status_code=status_code, raw_response=raw_response
        )
    if status_code == 408:
        return RequestTimeoutError(body, provider=provider, status_code=status_code)
    if status_code == 413:
        return ContextLengthError(
            body, provider=provider, status_code=status_code, raw_response=raw_response
        )
    if status_code == 429:
        retry_after: float | None = None
        if headers:
            ra_str = headers.get("retry-after") or headers.get("Retry-After")
            if ra_str:
                try:
                    retry_after = float(ra_str)
                except ValueError:
                    pass
        return RateLimitError(
            body,
            provider=provider,
            status_code=status_code,
            retry_after=retry_after,
            raw_response=raw_response,
        )
    if status_code >= 500:
        return ServerError(
            body, provider=provider, status_code=status_code, raw_response=raw_response
        )

    # --- Message-body classification (Spec §6.5) ---
    body_lower = body.lower()
    if "not found" in body_lower or "does not exist" in body_lower:
        return NotFoundError(
            body, provider=provider, status_code=status_code, raw_response=raw_response
        )
    if "context length" in body_lower or "too many tokens" in body_lower:
        return ContextLengthError(
            body, provider=provider, status_code=status_code, raw_response=raw_response
        )
    if "content filter" in body_lower or "safety" in body_lower:
        return ContentFilterError(
            body, provider=provider, status_code=status_code, raw_response=raw_response
        )
    if "unauthorized" in body_lower or "invalid key" in body_lower:
        return AuthenticationError(
            body, provider=provider, status_code=status_code, raw_response=raw_response
        )

    return ProviderError(
        body,
        provider=provider,
        status_code=status_code,
        retryable=True,
        raw_response=raw_response,
    )
