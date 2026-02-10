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
        super().__init__(
            message, provider=provider, status_code=status_code, retryable=retryable
        )
        self.raw_response = raw_response


class AuthenticationError(ProviderError):
    """401/403: Invalid or missing credentials. Not retryable."""

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
            retryable=False,
            raw_response=raw_response,
        )


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
            retryable=False,
            raw_response=raw_response,
        )


class InvalidRequestError(SDKError):
    """Bad request parameters. Not retryable."""

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
) -> ProviderError:
    """Map an HTTP status code to the appropriate error type. Spec §6.4."""
    if status_code in (401, 403):
        return AuthenticationError(
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
    if status_code == 408:
        return ProviderError(
            body, provider=provider, status_code=status_code, retryable=True,
            raw_response=raw_response,
        )
    if status_code >= 500:
        return ServerError(
            body, provider=provider, status_code=status_code, raw_response=raw_response
        )
    return ProviderError(
        body, provider=provider, status_code=status_code, retryable=False,
        raw_response=raw_response,
    )
