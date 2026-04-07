from rag.ingestion.base import Document, Ingestor

# Milestone 4: Implement web scraping using trafilatura.
# Set source_type="web", url=<url>.


class WebIngestor(Ingestor):
    def ingest(self, source: str) -> list[Document]:
        raise NotImplementedError("Web ingestion implemented in Milestone 4")
