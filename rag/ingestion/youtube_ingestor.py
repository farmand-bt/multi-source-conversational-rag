import hashlib
import json
import urllib.parse
import urllib.request
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from config.settings import CHUNK_SIZE
from rag.ingestion.base import Document, Ingestor


class YouTubeIngestor(Ingestor):
    """Fetches YouTube transcripts and returns timestamp-bounded Document chunks.

    Each returned Document covers ~CHUNK_SIZE characters of transcript text,
    preserving the start timestamp of the first segment in each group. This
    allows the Chunker to split further while keeping meaningful timestamps.
    """

    def __init__(self) -> None:
        self._api = YouTubeTranscriptApi()

    def ingest(self, source: str, source_name: str | None = None) -> list[Document]:
        """
        Args:
            source: YouTube URL (youtube.com/watch?v=… or youtu.be/…) or bare video ID.
            source_name: Override the display name; defaults to the video title.
        """
        video_id = _extract_video_id(source)
        url = f"https://www.youtube.com/watch?v={video_id}"

        try:
            # Try English first (covers manual + auto-generated English captions)
            transcript = self._api.fetch(video_id, languages=["en", "en-US", "en-GB"])
        except NoTranscriptFound:
            # Fall back to any available transcript, translating to English if possible
            try:
                transcript_list = self._api.list(video_id)
                first = next(iter(transcript_list), None)
                if first is None:
                    raise ValueError(f"No transcripts are available for video '{video_id}'.")
                try:
                    transcript = first.translate("en").fetch()
                except Exception:
                    transcript = first.fetch()  # use original language if translation fails
            except (VideoUnavailable, CouldNotRetrieveTranscript) as e:
                raise ValueError(f"Could not retrieve transcript for '{video_id}': {e}") from e
        except TranscriptsDisabled as e:
            raise ValueError(
                f"Transcripts have been disabled by the owner of video '{video_id}'."
            ) from e
        except (VideoUnavailable, CouldNotRetrieveTranscript) as e:
            raise ValueError(f"Could not retrieve transcript for '{video_id}': {e}") from e

        segments = [{"text": s.text.strip(), "start": s.start} for s in transcript]

        segments = [s for s in segments if s["text"]]
        if not segments:
            raise ValueError(f"Transcript is empty for video '{video_id}'.")

        title = source_name or _fetch_title(video_id) or video_id
        source_id = hashlib.sha256(video_id.encode()).hexdigest()[:16]

        return [
            Document(
                text=text,
                source_type="youtube",
                source_name=title,
                source_id=source_id,
                chunk_index=0,  # re-assigned by Chunker
                url=url,
                timestamp=timestamp,
            )
            for timestamp, text in _group_segments(segments)
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_video_id(source: str) -> str:
    """Extract the video ID from a URL or return the source as-is if it looks like an ID."""
    source = source.strip()
    if "youtu.be/" in source:
        return source.split("youtu.be/")[-1].split("?")[0].split("/")[0]
    parsed = urlparse(source)
    qs = parse_qs(parsed.query)
    if "v" in qs:
        return qs["v"][0]
    return source  # bare video ID


def _fetch_title(video_id: str) -> str | None:
    """Fetch the video title from the YouTube oEmbed endpoint (no API key required)."""
    video_url = urllib.parse.quote(f"https://www.youtube.com/watch?v={video_id}", safe="")
    oembed_url = f"https://www.youtube.com/oembed?url={video_url}&format=json"
    try:
        with urllib.request.urlopen(oembed_url, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return data.get("title")
    except Exception:
        return None


def _group_segments(segments: list[dict]) -> list[tuple[str, str]]:
    """Group transcript segments into (timestamp, text) pairs capped at ~CHUNK_SIZE chars.

    Grouping respects segment boundaries so timestamps remain meaningful.
    """
    groups: list[tuple[str, str]] = []
    buf: list[str] = []
    buf_len: int = 0
    group_start: float = 0.0

    for seg in segments:
        text = seg["text"]
        start = seg["start"]

        if not buf:
            group_start = start

        if buf_len + len(text) + 1 > CHUNK_SIZE and buf:
            groups.append((_format_ts(group_start), " ".join(buf)))
            buf = [text]
            group_start = start
            buf_len = len(text)
        else:
            buf.append(text)
            buf_len += len(text) + 1

    if buf:
        groups.append((_format_ts(group_start), " ".join(buf)))

    return groups


def _format_ts(seconds: float) -> str:
    """Convert seconds to MM:SS or H:MM:SS."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
