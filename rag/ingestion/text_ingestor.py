import hashlib

from rag.ingestion.base import Document, Ingestor


class TextIngestor(Ingestor):
    """Ingests plain text pasted directly by the user.

    Returns a single Document whose text is the full pasted content.
    The Chunker downstream splits it into properly-sized chunks.
    """

    def ingest(self, source: str | bytes, source_name: str | None = None) -> list[Document]:
        if isinstance(source, bytes):
            source = source.decode("utf-8", errors="replace")
        text = source.strip()
        if not text:
            raise ValueError("No text content provided.")
        name = (source_name or "Pasted text").strip()
        source_id = hashlib.sha256(text.encode()).hexdigest()[:16]
        return [
            Document(
                text=text,
                source_type="text",
                source_name=name,
                source_id=source_id,
                chunk_index=0,
            )
        ]
