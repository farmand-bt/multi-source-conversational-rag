from rag.chunking.chunker import Chunker
from rag.embeddings.embedder import Embedder
from rag.ingestion.pdf_ingestor import PDFIngestor
from rag.vectorstore.chroma_store import ChromaStore


class RAGPipeline:
    """
    Orchestrates the full RAG pipeline.

    Write path (M2):
        ingest() -> Ingestor -> Chunker -> Embedder -> ChromaStore

    Read path (M3+):
        query() -> Embedder.embed_query -> ChromaStore.query -> Generator
    """

    def __init__(self) -> None:
        self._embedder = Embedder()          # loads sentence-transformer model once
        self._chunker = Chunker()
        self._store = ChromaStore()
        self._pdf_ingestor = PDFIngestor()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def ingest(
        self,
        source: str | bytes,
        source_type: str,
        source_name: str | None = None,
    ) -> int:
        """Ingest a source into the vector store.

        Returns:
            Number of chunks stored.
        """
        if source_type == "pdf":
            docs = self._pdf_ingestor.ingest(source, source_name)
        else:
            raise NotImplementedError(
                f"Source type '{source_type}' is implemented in Milestone 4"
            )

        chunks = self._chunker.chunk(docs)
        embeddings = self._embedder.embed_documents([c.text for c in chunks])
        self._store.add(chunks, embeddings)
        return len(chunks)

    def delete_source(self, source_id: str) -> None:
        self._store.delete(source_id)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def list_sources(self) -> list[dict]:
        return self._store.list_sources()

    def query(self, text: str) -> str:
        # Milestone 3: embed query -> retrieve -> generate answer with citations
        raise NotImplementedError("Query pipeline implemented in Milestone 3")
