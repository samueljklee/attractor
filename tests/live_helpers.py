"""Shared helpers for live API tests.

Image construction, message builders, tool definitions, schemas, and
assertion helpers used by multiple live-test files.
"""

from __future__ import annotations

import struct
import zlib
from typing import Any

from attractor_llm import Message, Tool
from attractor_llm.types import ContentPart, ContentPartKind, ImageData, Role

# ------------------------------------------------------------------ #
# Image helpers
# ------------------------------------------------------------------ #

# Publicly accessible HTTP PNG -- fetched server-side by OpenAI/Anthropic.
# Gemini requires GCS/File API URIs, so the URL path is xfail for Gemini.
IMAGE_URL_HTTP = "https://httpbin.org/image/png"


def make_minimal_png(width: int = 8, height: int = 8) -> bytes:
    """Create a minimal valid RGB PNG image in memory (no Pillow required)."""

    def chunk(type_: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(type_ + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + type_ + data + struct.pack(">I", crc)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    idat = chunk(b"IDAT", zlib.compress(raw))
    iend = chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


PNG_BYTES = make_minimal_png()


def url_image_msg(prompt: str) -> Message:
    """User message carrying an HTTP image URL plus a text prompt."""
    return Message(
        role=Role.USER,
        content=[
            ContentPart(kind=ContentPartKind.TEXT, text=prompt),
            ContentPart(
                kind=ContentPartKind.IMAGE,
                image=ImageData(url=IMAGE_URL_HTTP, media_type="image/png"),
            ),
        ],
    )


def base64_image_msg(prompt: str) -> Message:
    """User message carrying inline base64 PNG bytes plus a text prompt."""
    return Message(
        role=Role.USER,
        content=[
            ContentPart(kind=ContentPartKind.TEXT, text=prompt),
            ContentPart(
                kind=ContentPartKind.IMAGE,
                image=ImageData(data=PNG_BYTES, media_type="image/png"),
            ),
        ],
    )


# ------------------------------------------------------------------ #
# Tool definitions
# ------------------------------------------------------------------ #


async def _weather_fn(city: str) -> str:
    return f"Weather in {city}: 22 C, sunny."


async def _time_fn(timezone: str) -> str:
    return f"Current time in {timezone}: 14:30 UTC."


async def _stock_fn(ticker: str) -> str:
    return f"Price of {ticker}: $42.00."


WEATHER_TOOL = Tool(
    name="get_weather",
    description="Get current weather for a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
    execute=_weather_fn,
)

TIME_TOOL = Tool(
    name="get_time",
    description="Get the current time in a timezone.",
    parameters={
        "type": "object",
        "properties": {"timezone": {"type": "string", "description": "Timezone name (e.g. UTC)"}},
        "required": ["timezone"],
    },
    execute=_time_fn,
)

STOCK_TOOL = Tool(
    name="get_stock_price",
    description="Get current stock price for a ticker symbol.",
    parameters={
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. AAPL)"}
        },
        "required": ["ticker"],
    },
    execute=_stock_fn,
)

# ------------------------------------------------------------------ #
# Shared schema
# ------------------------------------------------------------------ #

PERSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
}

# ------------------------------------------------------------------ #
# Assertion helpers
# ------------------------------------------------------------------ #


def assert_mostly_uppercase(text: str, *, threshold: float = 0.70) -> None:
    """Assert that at least *threshold* of alphabetic chars are uppercase."""
    letters = [c for c in text if c.isalpha()]
    assert letters, f"Expected alphabetic characters; got: {text!r}"
    ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    assert ratio >= threshold, f"Expected >={threshold:.0%} uppercase, got {ratio:.0%}: {text!r}"
