import hashlib
from urllib.parse import urlparse

import trafilatura
from trafilatura.metadata import extract_metadata

from rag.ingestion.base import Document, Ingestor


class WebIngestor(Ingestor):
    """Extracts clean article text from web pages using trafilatura."""

    def ingest(self, source: str, source_name: str | None = None) -> list[Document]:
        """
        Args:
            source: Full URL of the web page.
            source_name: Override the display name; defaults to the page title or URL.
        """
        url = source.strip()
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            raise ValueError(
                f"Could not download '{url}'. Check the URL and your internet connection."
            )

        text = trafilatura.extract(downloaded, include_tables=False, favor_recall=True)
        if not text:
            raise ValueError(f"No article text could be extracted from '{url}'.")

        meta = extract_metadata(downloaded, default_url=url)
        title = meta.title if meta and meta.title else None
        name = source_name or title or _display_url(url)
        source_id = hashlib.sha256(url.encode()).hexdigest()[:16]

        return [
            Document(
                text=text,
                source_type="web",
                source_name=name,
                source_id=source_id,
                chunk_index=0,
                url=url,
            )
        ]


def _display_url(url: str) -> str:
    """Return 'netloc/path' as a short human-readable fallback label."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    return f"{parsed.netloc}{path}" if path else parsed.netloc
