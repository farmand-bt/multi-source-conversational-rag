"""Tests for WebIngestor and YouTubeIngestor.

External HTTP calls (trafilatura.fetch_url, YouTubeTranscriptApi.fetch) are
mocked so tests run offline without touching real services.
"""

from unittest.mock import MagicMock, patch

import pytest

from rag.ingestion.web_ingestor import WebIngestor, _display_url
from rag.ingestion.youtube_ingestor import (
    YouTubeIngestor,
    _extract_video_id,
    _format_ts,
    _group_segments,
)


# ===========================================================================
# WebIngestor
# ===========================================================================

FAKE_HTML = b"<html><head><title>AI Overview</title></head><body><p>Text</p></body></html>"
FAKE_TEXT = "Artificial intelligence is a broad field of computer science."


def _mock_web(monkeypatch, text=FAKE_TEXT, title="AI Overview", downloaded=FAKE_HTML):
    monkeypatch.setattr("trafilatura.fetch_url", lambda url, **kw: downloaded)
    monkeypatch.setattr(
        "trafilatura.extract",
        lambda html, **kw: text,
    )
    meta_mock = MagicMock()
    meta_mock.title = title
    monkeypatch.setattr(
        "rag.ingestion.web_ingestor.extract_metadata",
        lambda html, **kw: meta_mock,
    )


class TestWebIngestor:
    def test_returns_single_document(self, monkeypatch):
        _mock_web(monkeypatch)
        docs = WebIngestor().ingest("https://example.com/article")
        assert len(docs) == 1

    def test_source_type_is_web(self, monkeypatch):
        _mock_web(monkeypatch)
        doc = WebIngestor().ingest("https://example.com")[0]
        assert doc.source_type == "web"

    def test_source_name_uses_page_title(self, monkeypatch):
        _mock_web(monkeypatch, title="My Article")
        doc = WebIngestor().ingest("https://example.com")[0]
        assert doc.source_name == "My Article"

    def test_source_name_falls_back_to_url_when_no_title(self, monkeypatch):
        _mock_web(monkeypatch, title=None)
        doc = WebIngestor().ingest("https://example.com/page")[0]
        assert "example.com" in doc.source_name

    def test_source_name_override(self, monkeypatch):
        _mock_web(monkeypatch)
        doc = WebIngestor().ingest("https://example.com", source_name="Custom Name")[0]
        assert doc.source_name == "Custom Name"

    def test_url_metadata_stored(self, monkeypatch):
        _mock_web(monkeypatch)
        url = "https://example.com/article"
        doc = WebIngestor().ingest(url)[0]
        assert doc.url == url

    def test_source_id_is_stable_for_same_url(self, monkeypatch):
        _mock_web(monkeypatch)
        url = "https://example.com/article"
        id1 = WebIngestor().ingest(url)[0].source_id
        id2 = WebIngestor().ingest(url)[0].source_id
        assert id1 == id2

    def test_source_id_differs_for_different_urls(self, monkeypatch):
        _mock_web(monkeypatch)
        id1 = WebIngestor().ingest("https://example.com/a")[0].source_id
        id2 = WebIngestor().ingest("https://example.com/b")[0].source_id
        assert id1 != id2

    def test_raises_if_download_fails(self, monkeypatch):
        monkeypatch.setattr("trafilatura.fetch_url", lambda url, **kw: None)
        with pytest.raises(ValueError, match="Could not download"):
            WebIngestor().ingest("https://example.com")

    def test_raises_if_extraction_fails(self, monkeypatch):
        monkeypatch.setattr("trafilatura.fetch_url", lambda url, **kw: FAKE_HTML)
        monkeypatch.setattr("trafilatura.extract", lambda html, **kw: None)
        with pytest.raises(ValueError, match="No article text"):
            WebIngestor().ingest("https://example.com")

    def test_chunk_index_initialized_to_zero(self, monkeypatch):
        _mock_web(monkeypatch)
        doc = WebIngestor().ingest("https://example.com")[0]
        assert doc.chunk_index == 0


class TestDisplayUrl:
    def test_returns_netloc_and_path(self):
        assert _display_url("https://example.com/article/one") == "example.com/article/one"

    def test_strips_trailing_slash(self):
        assert _display_url("https://example.com/") == "example.com"

    def test_bare_domain(self):
        assert _display_url("https://example.com") == "example.com"


# ===========================================================================
# YouTubeIngestor
# ===========================================================================

def _make_snippet(text: str, start: float, duration: float = 3.0) -> MagicMock:
    s = MagicMock()
    s.text = text
    s.start = start
    s.duration = duration
    return s


FAKE_SNIPPETS = [
    _make_snippet("Hello everyone welcome to this video", 0.0),
    _make_snippet("Today we are going to talk about Python", 3.0),
    _make_snippet("Python is a great language", 6.0),
]


def _mock_yt(monkeypatch, snippets=None, title="Python Tutorial"):
    if snippets is None:
        snippets = FAKE_SNIPPETS
    mock_api = MagicMock()
    mock_api.fetch.return_value = snippets
    monkeypatch.setattr(
        "rag.ingestion.youtube_ingestor.YouTubeTranscriptApi",
        lambda: mock_api,
    )
    monkeypatch.setattr(
        "rag.ingestion.youtube_ingestor._fetch_title",
        lambda video_id: title,
    )


class TestYouTubeIngestor:
    def test_returns_at_least_one_document(self, monkeypatch):
        _mock_yt(monkeypatch)
        docs = YouTubeIngestor().ingest("https://youtube.com/watch?v=dQw4w9WgXcQ")
        assert len(docs) >= 1

    def test_source_type_is_youtube(self, monkeypatch):
        _mock_yt(monkeypatch)
        doc = YouTubeIngestor().ingest("dQw4w9WgXcQ")[0]
        assert doc.source_type == "youtube"

    def test_source_name_uses_video_title(self, monkeypatch):
        _mock_yt(monkeypatch, title="My Video")
        doc = YouTubeIngestor().ingest("dQw4w9WgXcQ")[0]
        assert doc.source_name == "My Video"

    def test_source_name_override(self, monkeypatch):
        _mock_yt(monkeypatch)
        doc = YouTubeIngestor().ingest("dQw4w9WgXcQ", source_name="Custom")[0]
        assert doc.source_name == "Custom"

    def test_url_metadata_is_watch_url(self, monkeypatch):
        _mock_yt(monkeypatch)
        doc = YouTubeIngestor().ingest("dQw4w9WgXcQ")[0]
        assert doc.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_timestamp_metadata_set(self, monkeypatch):
        _mock_yt(monkeypatch)
        docs = YouTubeIngestor().ingest("dQw4w9WgXcQ")
        assert all(doc.timestamp is not None for doc in docs)

    def test_first_document_timestamp_is_zero(self, monkeypatch):
        _mock_yt(monkeypatch)
        doc = YouTubeIngestor().ingest("dQw4w9WgXcQ")[0]
        assert doc.timestamp == "0:00"

    def test_chunk_index_initialized_to_zero(self, monkeypatch):
        _mock_yt(monkeypatch)
        docs = YouTubeIngestor().ingest("dQw4w9WgXcQ")
        assert all(d.chunk_index == 0 for d in docs)

    def test_raises_on_transcripts_disabled(self, monkeypatch):
        from youtube_transcript_api._errors import TranscriptsDisabled

        mock_api = MagicMock()
        mock_api.fetch.side_effect = TranscriptsDisabled("abc")
        monkeypatch.setattr("rag.ingestion.youtube_ingestor.YouTubeTranscriptApi", lambda: mock_api)
        monkeypatch.setattr("rag.ingestion.youtube_ingestor._fetch_title", lambda v: None)
        with pytest.raises(ValueError, match="disabled"):
            YouTubeIngestor().ingest("abc")

    def test_source_id_stable_for_same_video(self, monkeypatch):
        _mock_yt(monkeypatch)
        id1 = YouTubeIngestor().ingest("dQw4w9WgXcQ")[0].source_id
        id2 = YouTubeIngestor().ingest("https://youtube.com/watch?v=dQw4w9WgXcQ")[0].source_id
        assert id1 == id2

    def test_falls_back_to_any_language_when_english_not_found(self, monkeypatch):
        """When no English transcript exists, fall back to any available language."""
        from youtube_transcript_api._errors import NoTranscriptFound

        french_snippet = _make_snippet("Bonjour tout le monde", 0.0)

        mock_transcript = MagicMock()
        mock_transcript.fetch.return_value = [french_snippet]

        mock_transcript_list = MagicMock()
        mock_transcript_list.__iter__ = MagicMock(return_value=iter([mock_transcript]))

        mock_api = MagicMock()
        # English fetch fails; list() returns a French transcript
        mock_api.fetch.side_effect = NoTranscriptFound("abc", ["en", "en-US", "en-GB"], [])
        mock_api.list.return_value = mock_transcript_list

        monkeypatch.setattr("rag.ingestion.youtube_ingestor.YouTubeTranscriptApi", lambda: mock_api)
        monkeypatch.setattr("rag.ingestion.youtube_ingestor._fetch_title", lambda v: "French Video")

        docs = YouTubeIngestor().ingest("abc")
        assert len(docs) >= 1
        assert docs[0].source_type == "youtube"
        assert "Bonjour" in docs[0].text


# ===========================================================================
# Unit helpers
# ===========================================================================

class TestExtractVideoId:
    def test_youtube_watch_url(self):
        assert _extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_youtu_be_url(self):
        assert _extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_youtu_be_with_params(self):
        assert _extract_video_id("https://youtu.be/dQw4w9WgXcQ?t=42") == "dQw4w9WgXcQ"

    def test_bare_video_id(self):
        assert _extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"


class TestFormatTs:
    def test_seconds_only(self):
        assert _format_ts(45.0) == "0:45"

    def test_minutes_and_seconds(self):
        assert _format_ts(154.0) == "2:34"

    def test_hours(self):
        assert _format_ts(3723.0) == "1:02:03"

    def test_zero(self):
        assert _format_ts(0.0) == "0:00"


class TestGroupSegments:
    def test_short_segments_fit_in_one_group(self):
        segs = [{"text": "Hello", "start": 0.0}, {"text": "World", "start": 1.0}]
        groups = _group_segments(segs)
        assert len(groups) == 1
        assert groups[0][0] == "0:00"
        assert "Hello" in groups[0][1]
        assert "World" in groups[0][1]

    def test_long_segments_split_into_multiple_groups(self):
        # Each segment is 100 chars; CHUNK_SIZE=500 so 6+ segments should split
        segs = [{"text": "a" * 100, "start": float(i * 10)} for i in range(8)]
        groups = _group_segments(segs)
        assert len(groups) >= 2

    def test_timestamp_matches_first_segment_in_group(self):
        segs = [{"text": "word ", "start": 0.0}, {"text": "word ", "start": 5.0}]
        groups = _group_segments(segs)
        assert groups[0][0] == "0:00"

    def test_empty_segments_returns_empty(self):
        assert _group_segments([]) == []
