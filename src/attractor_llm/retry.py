"""Retry engine with exponential backoff for the Unified LLM Client SDK.

Implements retry logic from the Unified LLM Client Specification §6.6.
Honors provider Retry-After headers when available.
"""

from __future__ import annotations

import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

import anyio

from .errors import RateLimitError, SDKError

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Configuration for retry behavior with exponential backoff.

    The jitter strategy is "equal jitter": delay is uniformly distributed
    in [0.5 * computed_delay, 1.0 * computed_delay]. This prevents
    thundering-herd effects while keeping delays reasonably close to
    the computed backoff.
    """

    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True

    def compute_delay(self, attempt: int) -> float:
        """Compute delay for a given attempt number (0-indexed)."""
        delay = self.initial_delay * (self.backoff_factor**attempt)
        delay = min(delay, self.max_delay)
        if self.jitter:
            # Equal jitter: uniform in [delay/2, delay]
            delay = delay * (0.5 + random.random() * 0.5)  # noqa: S311
        return delay


OnRetryCallback = Callable[[int, SDKError, float], Awaitable[None] | None]


async def retry_with_policy(
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    on_retry: OnRetryCallback | None = None,
) -> T:
    """Execute an async function with retry logic.

    Only retries errors where ``error.retryable`` is True.
    Uses exponential backoff with optional jitter.
    Honors ``RateLimitError.retry_after`` when present (uses the larger
    of computed backoff and server-requested delay).

    Args:
        fn: Async callable to execute.
        policy: Retry configuration.
        on_retry: Optional callback invoked before each retry with
            (attempt, error, delay).

    Returns:
        The result of a successful ``fn()`` call.

    Raises:
        SDKError: When retries are exhausted or error is not retryable.
        ValueError: If policy.max_retries is negative.
    """
    if policy.max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {policy.max_retries}")

    last_error: SDKError | None = None

    for attempt in range(policy.max_retries + 1):
        try:
            return await fn()
        except SDKError as exc:
            last_error = exc

            if not exc.retryable:
                raise

            if attempt >= policy.max_retries:
                raise

            delay = policy.compute_delay(attempt)

            # Honor Retry-After from rate limit responses (Spec §6.6)
            if isinstance(exc, RateLimitError) and exc.retry_after is not None:
                delay = max(delay, exc.retry_after)

            if on_retry is not None:
                result = on_retry(attempt, exc, delay)
                if isinstance(result, Awaitable):
                    await result

            await anyio.sleep(delay)

    # Should be unreachable, but satisfies the type checker.
    assert last_error is not None  # noqa: S101
    raise last_error
