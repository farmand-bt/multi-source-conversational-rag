from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Document:
    """A chunk of text with its source metadata."""

    text: str
    source_type: str          # "pdf" | "web" | "youtube"
    source_name: str          # human-readable name (filename, page title, video title)
    source_id: str            # stable hash used for deletion/deduplication
    chunk_index: int          # globally sequential index assigned by Chunker across all chunks in a batch
    ingested_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    # Source-type-specific location fields — set whichever applies, leave others None
    page_number: int | None = None   # PDF
    url: str | None = None           # web / youtube
    timestamp: str | None = None     # youtube (e.g. "2:34")


class Ingestor(ABC):
    """Abstract base class for all source ingestors.

    Each subclass handles one source type (PDF, web URL, YouTube).
    Implement `ingest` to return a flat list of Document objects with
    the full metadata schema populated — chunking happens downstream.
    """

    @abstractmethod
    def ingest(self, source: str | bytes, source_name: str | None = None) -> list[Document]:
        """Extract raw text from `source` and return as Document objects.

        Args:
            source: File path / URL (str) or raw bytes (e.g. from Streamlit uploader).
            source_name: Human-readable label; inferred from path/URL if not given.

        Returns:
            List of Document objects, one per logical unit (e.g. one per PDF
            page), before chunking. chunk_index is 0 on all returned documents;
            the Chunker re-assigns indices globally across the batch.
        """
        ...
