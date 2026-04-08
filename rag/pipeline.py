from rag.chunking.chunker import Chunker
from rag.embeddings.embedder import Embedder
from rag.generation.generator import Generator
from rag.ingestion.pdf_ingestor import PDFIngestor
from rag.models import Answer
from rag.retrieval.retriever import Retriever
from rag.vectorstore.chroma_store import ChromaStore


class RAGPipeline:
    """
    Orchestrates the full RAG pipeline.

    Write path (M2):
        ingest() -> Ingestor -> Chunker -> Embedder -> ChromaStore

    Read path (M3):
        ask() -> Retriever -> Generator -> Answer
    """

    def __init__(self) -> None:
        self._embedder = Embedder()          # loads sentence-transformer model once
        self._chunker = Chunker()
        self._store = ChromaStore()
        self._pdf_ingestor = PDFIngestor()
        self._retriever = Retriever(self._embedder, self._store)
        self._generator: Generator | None = None  # lazy: avoids startup failure without GWDG creds

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

    def ask(self, query: str, history: list[dict] | None = None) -> Answer:
        """Answer a question using retrieved context.

        Returns an Answer with no citations if the store is empty or query is blank.
        """
        results = self._retriever.retrieve(query)
        if not results:
            return Answer(text="I don't have enough information to answer that question.")

        context_docs = [doc for doc, _ in results]
        raw = self._get_generator().generate(query, context_docs, history)
        return Answer.from_raw(raw)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_generator(self) -> Generator:
        if self._generator is None:
            self._generator = Generator()
        return self._generator
