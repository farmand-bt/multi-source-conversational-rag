from rag.ingestion.base import Document, Ingestor

# Milestone 4: Implement YouTube transcript fetching using youtube-transcript-api.
# Chunk by timestamp-boundary segments when possible (not just character count).
# Set source_type="youtube", url=<url>, timestamp=<"MM:SS">.


class YouTubeIngestor(Ingestor):
    def ingest(self, source: str) -> list[Document]:
        raise NotImplementedError("YouTube ingestion implemented in Milestone 4")
