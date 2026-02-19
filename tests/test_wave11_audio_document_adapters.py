"""Tests for Wave 11 P3 – audio and document adapter handling (§3.3-3.5, §8.3).

Support matrix:
  - Gemini:    AUDIO → inlineData / fileData,  DOCUMENT → inlineData / fileData
  - Anthropic: AUDIO → raise InvalidRequestError, DOCUMENT → base64 document source
  - OpenAI:    AUDIO → raise InvalidRequestError, DOCUMENT → raise InvalidRequestError
  - OAI-Compat: AUDIO → raise InvalidRequestError, DOCUMENT → raise InvalidRequestError
"""

from __future__ import annotations

import base64

import pytest

from attractor_llm.adapters.anthropic import AnthropicAdapter
from attractor_llm.adapters.base import ProviderConfig
from attractor_llm.adapters.gemini import GeminiAdapter
from attractor_llm.adapters.openai import OpenAIAdapter
from attractor_llm.adapters.openai_compat import OpenAICompatAdapter
from attractor_llm.errors import InvalidRequestError
from attractor_llm.types import (
    AudioData,
    ContentPart,
    ContentPartKind,
    DocumentData,
    Message,
    Request,
    Role,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_CONFIG = ProviderConfig(api_key="test-key")

AUDIO_BYTES = b"RIFF\x00\x00\x00\x00WAVEfmt "
DOCUMENT_BYTES = b"%PDF-1.4 test document bytes"

AUDIO_DATA_INLINE = AudioData(data=AUDIO_BYTES, media_type="audio/wav")
AUDIO_DATA_URL = AudioData(url="gs://bucket/audio.wav", media_type="audio/mpeg")
DOCUMENT_DATA_INLINE = DocumentData(data=DOCUMENT_BYTES, media_type="application/pdf")
DOCUMENT_DATA_URL = DocumentData(url="gs://bucket/doc.pdf", media_type="application/pdf")

AUDIO_PART_INLINE = ContentPart.audio_part(AUDIO_DATA_INLINE)
AUDIO_PART_URL = ContentPart.audio_part(AUDIO_DATA_URL)
DOCUMENT_PART_INLINE = ContentPart.document_part(DOCUMENT_DATA_INLINE)
DOCUMENT_PART_URL = ContentPart.document_part(DOCUMENT_DATA_URL)


def _user_msg(*parts: ContentPart) -> Message:
    return Message(role=Role.USER, content=list(parts))


def _make_request(*parts: ContentPart) -> Request:
    return Request(model="test-model", messages=[_user_msg(*parts)])


# ---------------------------------------------------------------------------
# Gemini adapter tests
# ---------------------------------------------------------------------------


class TestGeminiAudio:
    """Gemini translates audio to inlineData / fileData."""

    def setup_method(self) -> None:
        self.adapter = GeminiAdapter(_CONFIG)

    def test_audio_data_translates_to_inline_data(self) -> None:
        result = self.adapter._translate_part(AUDIO_PART_INLINE)
        assert result is not None
        assert "inlineData" in result
        inline = result["inlineData"]
        assert inline["mimeType"] == "audio/wav"
        # Verify base64 round-trips correctly
        assert base64.b64decode(inline["data"]) == AUDIO_BYTES

    def test_audio_url_translates_to_file_data(self) -> None:
        result = self.adapter._translate_part(AUDIO_PART_URL)
        assert result is not None
        assert "fileData" in result
        fd = result["fileData"]
        assert fd["mimeType"] == "audio/mpeg"
        assert fd["fileUri"] == "gs://bucket/audio.wav"

    def test_audio_default_mime_type_when_unset(self) -> None:
        """Falls back to audio/wav when media_type is None."""
        part = ContentPart.audio_part(AudioData(data=b"bytes"))
        result = self.adapter._translate_part(part)
        assert result is not None
        assert result["inlineData"]["mimeType"] == "audio/wav"


class TestGeminiDocument:
    """Gemini translates documents to inlineData / fileData."""

    def setup_method(self) -> None:
        self.adapter = GeminiAdapter(_CONFIG)

    def test_document_data_translates_to_inline_data(self) -> None:
        result = self.adapter._translate_part(DOCUMENT_PART_INLINE)
        assert result is not None
        assert "inlineData" in result
        inline = result["inlineData"]
        assert inline["mimeType"] == "application/pdf"
        assert base64.b64decode(inline["data"]) == DOCUMENT_BYTES

    def test_document_url_translates_to_file_data(self) -> None:
        result = self.adapter._translate_part(DOCUMENT_PART_URL)
        assert result is not None
        assert "fileData" in result
        fd = result["fileData"]
        assert fd["mimeType"] == "application/pdf"
        assert fd["fileUri"] == "gs://bucket/doc.pdf"

    def test_document_default_mime_type_when_unset(self) -> None:
        """Falls back to application/pdf when media_type is None."""
        part = ContentPart.document_part(DocumentData(data=b"%PDF"))
        result = self.adapter._translate_part(part)
        assert result is not None
        assert result["inlineData"]["mimeType"] == "application/pdf"

    def test_gemini_audio_document_end_to_end_in_translate_contents(self) -> None:
        """Audio + document parts survive the full message translation pipeline."""
        msg = _user_msg(
            ContentPart.text_part("Analyse this"),
            AUDIO_PART_INLINE,
            DOCUMENT_PART_INLINE,
        )
        contents = self.adapter._translate_contents([msg])
        assert len(contents) == 1
        parts = contents[0]["parts"]
        # text + inlineData (audio) + inlineData (document) = 3 parts
        assert len(parts) == 3
        assert parts[0] == {"text": "Analyse this"}
        assert "inlineData" in parts[1]
        assert "inlineData" in parts[2]


# ---------------------------------------------------------------------------
# Anthropic adapter tests
# ---------------------------------------------------------------------------


class TestAnthropicDocument:
    """Anthropic translates documents to base64 document source."""

    def setup_method(self) -> None:
        self.adapter = AnthropicAdapter(_CONFIG)

    def test_document_data_translates_to_base64_source(self) -> None:
        result = self.adapter._translate_content_part(DOCUMENT_PART_INLINE, Role.USER)
        assert result["type"] == "document"
        source = result["source"]
        assert source["type"] == "base64"
        assert source["media_type"] == "application/pdf"
        assert base64.b64decode(source["data"]) == DOCUMENT_BYTES

    def test_document_default_mime_type_when_unset(self) -> None:
        part = ContentPart.document_part(DocumentData(data=b"%PDF"))
        result = self.adapter._translate_content_part(part, Role.USER)
        assert result["source"]["media_type"] == "application/pdf"

    def test_document_url_raises_invalid_request_error(self) -> None:
        """URL-only documents are not supported by Anthropic -- must raise."""
        with pytest.raises(InvalidRequestError, match="base64"):
            self.adapter._translate_content_part(DOCUMENT_PART_URL, Role.USER)


class TestAnthropicAudio:
    """Anthropic raises InvalidRequestError for audio content."""

    def setup_method(self) -> None:
        self.adapter = AnthropicAdapter(_CONFIG)

    def test_audio_raises_invalid_request_error(self) -> None:
        with pytest.raises(InvalidRequestError, match="audio"):
            self.adapter._translate_content_part(AUDIO_PART_INLINE, Role.USER)

    def test_audio_url_raises_invalid_request_error(self) -> None:
        with pytest.raises(InvalidRequestError, match="audio"):
            self.adapter._translate_content_part(AUDIO_PART_URL, Role.USER)

    def test_invalid_request_error_provider_is_anthropic(self) -> None:
        with pytest.raises(InvalidRequestError) as exc_info:
            self.adapter._translate_content_part(AUDIO_PART_INLINE, Role.USER)
        assert exc_info.value.provider == "anthropic"


# ---------------------------------------------------------------------------
# OpenAI adapter tests
# ---------------------------------------------------------------------------


class TestOpenAIAudio:
    """OpenAI raises InvalidRequestError for audio content."""

    def setup_method(self) -> None:
        self.adapter = OpenAIAdapter(_CONFIG)

    def test_audio_raises_invalid_request_error(self) -> None:
        msg = _user_msg(ContentPart.text_part("Listen:"), AUDIO_PART_INLINE)
        with pytest.raises(InvalidRequestError, match="audio"):
            self.adapter._translate_user_content(msg)

    def test_audio_url_raises_invalid_request_error(self) -> None:
        msg = _user_msg(AUDIO_PART_URL)
        with pytest.raises(InvalidRequestError, match="audio"):
            self.adapter._translate_user_content(msg)

    def test_invalid_request_error_provider_is_openai(self) -> None:
        msg = _user_msg(AUDIO_PART_INLINE)
        with pytest.raises(InvalidRequestError) as exc_info:
            self.adapter._translate_user_content(msg)
        assert exc_info.value.provider == "openai"


class TestOpenAIDocument:
    """OpenAI raises InvalidRequestError for document content."""

    def setup_method(self) -> None:
        self.adapter = OpenAIAdapter(_CONFIG)

    def test_document_raises_invalid_request_error(self) -> None:
        msg = _user_msg(DOCUMENT_PART_INLINE)
        with pytest.raises(InvalidRequestError, match="document"):
            self.adapter._translate_user_content(msg)

    def test_document_url_raises_invalid_request_error(self) -> None:
        msg = _user_msg(DOCUMENT_PART_URL)
        with pytest.raises(InvalidRequestError, match="document"):
            self.adapter._translate_user_content(msg)

    def test_invalid_request_error_provider_is_openai(self) -> None:
        msg = _user_msg(DOCUMENT_PART_INLINE)
        with pytest.raises(InvalidRequestError) as exc_info:
            self.adapter._translate_user_content(msg)
        assert exc_info.value.provider == "openai"


class TestOpenAIUnknownPartKindRaises:
    """The OpenAI default case now raises instead of returning placeholder text."""

    def setup_method(self) -> None:
        self.adapter = OpenAIAdapter(_CONFIG)

    def test_no_placeholder_text_for_audio(self) -> None:
        """Ensure no '[unsupported: audio]' text leaks through."""
        msg = _user_msg(AUDIO_PART_INLINE)
        with pytest.raises(InvalidRequestError):
            self.adapter._translate_user_content(msg)

    def test_no_placeholder_text_for_document(self) -> None:
        """Ensure no '[unsupported: document]' text leaks through."""
        msg = _user_msg(DOCUMENT_PART_INLINE)
        with pytest.raises(InvalidRequestError):
            self.adapter._translate_user_content(msg)


# ---------------------------------------------------------------------------
# OpenAI-compat adapter tests
# ---------------------------------------------------------------------------


class TestOpenAICompatAudio:
    """OpenAI-compat raises InvalidRequestError for audio content."""

    def setup_method(self) -> None:
        self.adapter = OpenAICompatAdapter(_CONFIG)

    def test_audio_raises_invalid_request_error(self) -> None:
        request = _make_request(AUDIO_PART_INLINE)
        with pytest.raises(InvalidRequestError, match="audio"):
            self.adapter._build_request_body(request)

    def test_audio_url_raises_invalid_request_error(self) -> None:
        request = _make_request(AUDIO_PART_URL)
        with pytest.raises(InvalidRequestError, match="audio"):
            self.adapter._build_request_body(request)

    def test_invalid_request_error_provider_is_openai_compat(self) -> None:
        request = _make_request(AUDIO_PART_INLINE)
        with pytest.raises(InvalidRequestError) as exc_info:
            self.adapter._build_request_body(request)
        assert exc_info.value.provider == "openai-compat"


class TestOpenAICompatDocument:
    """OpenAI-compat raises InvalidRequestError for document content."""

    def setup_method(self) -> None:
        self.adapter = OpenAICompatAdapter(_CONFIG)

    def test_document_raises_invalid_request_error(self) -> None:
        request = _make_request(DOCUMENT_PART_INLINE)
        with pytest.raises(InvalidRequestError, match="document"):
            self.adapter._build_request_body(request)

    def test_document_url_raises_invalid_request_error(self) -> None:
        request = _make_request(DOCUMENT_PART_URL)
        with pytest.raises(InvalidRequestError, match="document"):
            self.adapter._build_request_body(request)

    def test_invalid_request_error_provider_is_openai_compat(self) -> None:
        request = _make_request(DOCUMENT_PART_INLINE)
        with pytest.raises(InvalidRequestError) as exc_info:
            self.adapter._build_request_body(request)
        assert exc_info.value.provider == "openai-compat"


# ---------------------------------------------------------------------------
# Cross-adapter: no silent degradation
# ---------------------------------------------------------------------------


class TestNoMoreSilentDegradation:
    """Verify no adapter silently returns placeholder text for AUDIO or DOCUMENT.

    Each adapter must either properly translate or raise -- never emit
    '[unsupported: audio]', '[unsupported: document]', or any other
    silent-fallback placeholder for these two first-class content types.
    """

    PLACEHOLDER_SIGNALS = ("[unsupported:", "[audio", "[document")

    def _contains_placeholder(self, value: object) -> bool:
        """Recursively check if any string in value looks like a placeholder."""
        if isinstance(value, str):
            return any(sig in value for sig in self.PLACEHOLDER_SIGNALS)
        if isinstance(value, dict):
            return any(self._contains_placeholder(v) for v in value.values())
        if isinstance(value, list):
            return any(self._contains_placeholder(item) for item in value)
        return False

    # -- Gemini: should translate (not raise, not placeholder) --

    def test_gemini_audio_no_placeholder(self) -> None:
        adapter = GeminiAdapter(_CONFIG)
        result = adapter._translate_part(AUDIO_PART_INLINE)
        assert not self._contains_placeholder(result), f"Gemini silently degraded audio: {result}"

    def test_gemini_document_no_placeholder(self) -> None:
        adapter = GeminiAdapter(_CONFIG)
        result = adapter._translate_part(DOCUMENT_PART_INLINE)
        assert not self._contains_placeholder(result), (
            f"Gemini silently degraded document: {result}"
        )

    # -- Anthropic: should raise or translate properly (document) --

    def test_anthropic_audio_no_placeholder(self) -> None:
        adapter = AnthropicAdapter(_CONFIG)
        with pytest.raises(InvalidRequestError):
            adapter._translate_content_part(AUDIO_PART_INLINE, Role.USER)

    def test_anthropic_document_no_placeholder(self) -> None:
        adapter = AnthropicAdapter(_CONFIG)
        result = adapter._translate_content_part(DOCUMENT_PART_INLINE, Role.USER)
        assert not self._contains_placeholder(result), (
            f"Anthropic silently degraded document: {result}"
        )

    # -- OpenAI: should raise --

    def test_openai_audio_no_placeholder(self) -> None:
        adapter = OpenAIAdapter(_CONFIG)
        msg = _user_msg(AUDIO_PART_INLINE)
        with pytest.raises(InvalidRequestError):
            adapter._translate_user_content(msg)

    def test_openai_document_no_placeholder(self) -> None:
        adapter = OpenAIAdapter(_CONFIG)
        msg = _user_msg(DOCUMENT_PART_INLINE)
        with pytest.raises(InvalidRequestError):
            adapter._translate_user_content(msg)

    # -- OpenAI-compat: should raise --

    def test_openai_compat_audio_no_placeholder(self) -> None:
        adapter = OpenAICompatAdapter(_CONFIG)
        request = _make_request(AUDIO_PART_INLINE)
        with pytest.raises(InvalidRequestError):
            adapter._build_request_body(request)

    def test_openai_compat_document_no_placeholder(self) -> None:
        adapter = OpenAICompatAdapter(_CONFIG)
        request = _make_request(DOCUMENT_PART_INLINE)
        with pytest.raises(InvalidRequestError):
            adapter._build_request_body(request)

    # -- Anthropic case _: no longer emits placeholder --

    def test_anthropic_default_case_raises_not_placeholder(self) -> None:
        """The Anthropic fallback case now raises instead of returning text."""
        # We test this via a AUDIO part (which hits an explicit case, not _:, but
        # confirms the overall philosophy: no silent degradation anywhere)
        adapter = AnthropicAdapter(_CONFIG)
        with pytest.raises(InvalidRequestError):
            adapter._translate_content_part(AUDIO_PART_INLINE, Role.USER)
        # If we get here without raising, something is wrong


# ---------------------------------------------------------------------------
# Integration-style: content parts survive full request translation
# ---------------------------------------------------------------------------


class TestGeminiFullRequestTranslation:
    """Audio/document parts integrate cleanly into the full Gemini request body."""

    def setup_method(self) -> None:
        self.adapter = GeminiAdapter(_CONFIG)

    def test_audio_in_request_body(self) -> None:
        request = Request(
            model="gemini-2.0-flash",
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        ContentPart.text_part("Transcribe this audio:"),
                        AUDIO_PART_INLINE,
                    ],
                )
            ],
        )
        body = self.adapter._translate_request(request)
        contents = body["contents"]
        assert len(contents) == 1
        parts = contents[0]["parts"]
        assert parts[0] == {"text": "Transcribe this audio:"}
        assert "inlineData" in parts[1]
        assert parts[1]["inlineData"]["mimeType"] == "audio/wav"

    def test_document_in_request_body(self) -> None:
        request = Request(
            model="gemini-2.0-flash",
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        ContentPart.text_part("Summarise this PDF:"),
                        DOCUMENT_PART_INLINE,
                    ],
                )
            ],
        )
        body = self.adapter._translate_request(request)
        parts = body["contents"][0]["parts"]
        assert "inlineData" in parts[1]
        assert parts[1]["inlineData"]["mimeType"] == "application/pdf"
        assert base64.b64decode(parts[1]["inlineData"]["data"]) == DOCUMENT_BYTES


# ---------------------------------------------------------------------------
# Edge-case tests: None / empty payloads raise InvalidRequestError
# ---------------------------------------------------------------------------


class TestGeminiAudioEdgeCases:
    """Gemini raises on None audio payload or AudioData with no data and no url."""

    def setup_method(self) -> None:
        self.adapter = GeminiAdapter(_CONFIG)

    def test_gemini_audio_none_payload_raises(self) -> None:
        """AUDIO part with audio=None must raise, not silently return None.

        Uses model_construct to bypass Pydantic validation so we can reach
        the adapter's defensive guard directly.
        """
        part = ContentPart.model_construct(kind=ContentPartKind.AUDIO, audio=None)
        with pytest.raises(InvalidRequestError, match="no audio payload"):
            self.adapter._translate_part(part)

    def test_gemini_audio_no_data_no_url_raises(self) -> None:
        """AudioData with neither data nor url must raise.

        Uses model_construct to bypass Pydantic validation on AudioData.
        """
        audio = AudioData.model_construct(data=None, url=None, media_type="audio/wav")
        part = ContentPart.model_construct(kind=ContentPartKind.AUDIO, audio=audio)
        with pytest.raises(InvalidRequestError, match="no data and no url"):
            self.adapter._translate_part(part)


class TestGeminiDocumentEdgeCases:
    """Gemini raises on None document payload or DocumentData with no data and no url."""

    def setup_method(self) -> None:
        self.adapter = GeminiAdapter(_CONFIG)

    def test_gemini_document_none_payload_raises(self) -> None:
        """DOCUMENT part with document=None must raise, not silently return None.

        Uses model_construct to bypass Pydantic validation so we can reach
        the adapter's defensive guard directly.
        """
        part = ContentPart.model_construct(kind=ContentPartKind.DOCUMENT, document=None)
        with pytest.raises(InvalidRequestError, match="no document payload"):
            self.adapter._translate_part(part)

    def test_gemini_document_no_data_no_url_raises(self) -> None:
        """DocumentData with neither data nor url must raise.

        Uses model_construct to bypass Pydantic validation on DocumentData.
        """
        doc = DocumentData.model_construct(data=None, url=None, media_type="application/pdf")
        part = ContentPart.model_construct(kind=ContentPartKind.DOCUMENT, document=doc)
        with pytest.raises(InvalidRequestError, match="no data and no url"):
            self.adapter._translate_part(part)


class TestAnthropicDocumentEdgeCases:
    """Anthropic raises on None document payload."""

    def setup_method(self) -> None:
        self.adapter = AnthropicAdapter(_CONFIG)

    def test_anthropic_document_none_payload_raises(self) -> None:
        """DOCUMENT part with document=None must raise with a clear message.

        Uses model_construct to bypass Pydantic validation so we can reach
        the adapter's defensive guard directly.
        """
        part = ContentPart.model_construct(kind=ContentPartKind.DOCUMENT, document=None)
        with pytest.raises(InvalidRequestError, match="no document payload"):
            self.adapter._translate_content_part(part, Role.USER)
