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
    chunk_index: int          # position of this chunk within the source
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
    def ingest(self, source: str) -> list[Document]:
        """Extract raw text from `source` and return as Document objects.

        Args:
            source: File path (PDF) or URL (web / YouTube).

        Returns:
            List of Document objects, one per logical unit (e.g. one per
            PDF page), before chunking. Chunk index should be 0 for all
            pre-chunk documents; the chunker will re-assign indices.
        """
        ...
