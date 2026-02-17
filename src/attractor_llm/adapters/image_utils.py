"""Shared image utilities for provider adapters.

Handles local file path detection and resolution for ImageData URLs,
per unified-llm-spec §3.5.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from attractor_llm.types import ImageData


def resolve_image_data(image: ImageData) -> ImageData:
    """Resolve local file paths in ImageData.url to inline data.

    Per spec §3.5: when url looks like a local file path (starts with
    ``/``, ``./``, or ``~``), the adapter reads the file, infers MIME type,
    and base64-encodes. The returned ImageData has ``data`` and
    ``media_type`` populated; ``url`` is cleared.

    If the url is not a local path (e.g., https://), the image is returned
    unchanged. If ``data`` is already set, the image is returned as-is.
    """
    # Already has inline data -- nothing to do
    if image.data is not None:
        return image

    url = image.url
    if url is None:
        return image

    # Detect local file path patterns
    if not (url.startswith("/") or url.startswith("./") or url.startswith("~")):
        return image

    # Read file and resolve
    p = Path(url).expanduser().resolve()
    if not p.exists():
        # Return unchanged if file doesn't exist -- adapter will handle the error
        return image

    mime, _ = mimetypes.guess_type(str(p))
    if mime is None or not mime.startswith("image/"):
        mime = "application/octet-stream"

    raw = p.read_bytes()
    return ImageData(data=raw, media_type=mime)
