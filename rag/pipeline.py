from rag.chunking.chunker import Chunker
from rag.embeddings.embedder import Embedder
from rag.generation.generator import Generator
from rag.ingestion.pdf_ingestor import PDFIngestor
from rag.ingestion.web_ingestor import WebIngestor
from rag.ingestion.youtube_ingestor import YouTubeIngestor
from rag.memory.conversation import ConversationMemory
from rag.models import Answer
from rag.retrieval.retriever import Retriever
from rag.vectorstore.chroma_store import ChromaStore


class RAGPipeline:
    """
    Orchestrates the full RAG pipeline.

    Write path (M2–M4):
        ingest() -> Ingestor -> Chunker -> Embedder -> ChromaStore

    Read path (M3–M5):
        ask() -> [ConversationMemory rewrite] -> Retriever -> Generator -> Answer
    """

    def __init__(self) -> None:
        self._embedder = Embedder()          # loads sentence-transformer model once
        self._chunker = Chunker()
        self._store = ChromaStore()
        self._pdf_ingestor = PDFIngestor()
        self._web_ingestor = WebIngestor()
        self._yt_ingestor = YouTubeIngestor()
        self._retriever = Retriever(self._embedder, self._store)
        self._generator: Generator | None = None  # lazy: avoids startup failure without GWDG creds
        self._memory = ConversationMemory()

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
        elif source_type == "web":
            docs = self._web_ingestor.ingest(source, source_name)
        elif source_type == "youtube":
            docs = self._yt_ingestor.ingest(source, source_name)
        else:
            raise ValueError(f"Unknown source type: '{source_type}'")

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

        If *history* is provided and non-empty, the query is first rewritten
        into a standalone question to improve retrieval accuracy. The original
        query (with full history) is still sent to the generator so the LLM
        has complete conversational context when forming its answer.

        Returns an Answer with no citations if the store is empty or query is blank.
        """
        # Rewrite follow-up questions so retrieval finds the right chunks
        if history:
            rewritten = self._memory.rewrite_query(query, history)
        else:
            rewritten = query

        results = self._retriever.retrieve(rewritten)
        if not results:
            return Answer(text="I don't have enough information to answer that question.")

        context_docs = [doc for doc, _ in results]
        # Generator receives the *original* query + full history so it can
        # answer in the conversational context (not the rewritten standalone form).
        raw = self._get_generator().generate(query, context_docs, history)
        rewritten_for_display = rewritten if rewritten != query else ""
        return Answer.from_raw(raw, rewritten_query=rewritten_for_display)

    def clear_history(self) -> None:
        """Clear the conversation memory (called when the user resets the chat)."""
        self._memory.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_generator(self) -> Generator:
        if self._generator is None:
            self._generator = Generator()
        return self._generator
