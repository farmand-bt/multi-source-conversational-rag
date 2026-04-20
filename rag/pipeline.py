from collections.abc import Iterator

from config.settings import MAX_CHUNKS_PER_SOURCE, TOP_K
from rag.chunking.chunker import Chunker
from rag.embeddings.embedder import Embedder
from rag.generation.generator import Generator
from rag.ingestion.arxiv_ingestor import ArXivIngestor
from rag.ingestion.pdf_ingestor import PDFIngestor
from rag.ingestion.text_ingestor import TextIngestor
from rag.ingestion.web_ingestor import WebIngestor
from rag.ingestion.youtube_ingestor import YouTubeIngestor
from rag.memory.conversation import ConversationMemory
from rag.models import Answer
from rag.retrieval.bm25_index import BM25Index
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

    def __init__(
        self,
        embedder: Embedder | None = None,
        ephemeral: bool = False,
        collection_name: str = "documents",
    ) -> None:
        self._embedder = embedder or Embedder()  # accepts a pre-loaded shared embedder
        self._chunker = Chunker()
        self._store = ChromaStore(ephemeral=ephemeral, collection_name=collection_name)
        self._pdf_ingestor = PDFIngestor()
        self._web_ingestor = WebIngestor()
        self._yt_ingestor = YouTubeIngestor()
        self._arxiv_ingestor = ArXivIngestor()
        self._text_ingestor = TextIngestor()
        self._bm25_index = BM25Index()
        self._retriever = Retriever(self._embedder, self._store, self._bm25_index)
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
        elif source_type == "arxiv":
            docs = self._arxiv_ingestor.ingest(source, source_name)
        elif source_type == "text":
            docs = self._text_ingestor.ingest(source, source_name)
        else:
            raise ValueError(f"Unknown source type: '{source_type}'")

        chunks = self._chunker.chunk(docs)
        embeddings = self._embedder.embed_documents([c.text for c in chunks])
        self._store.add(chunks, embeddings)
        self._bm25_index.add(chunks)
        return len(chunks)

    def delete_source(self, source_id: str) -> None:
        self._store.delete(source_id)
        self._bm25_index.delete(source_id)

    def delete_all_sources(self) -> None:
        """Remove every ingested chunk from the vector store and keyword index."""
        self._store.delete_all()
        self._bm25_index.delete_all()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def list_sources(self) -> list[dict]:
        return self._store.list_sources()

    # ------------------------------------------------------------------
    # Granular read-path steps (used by the UI for live progress display)
    # ------------------------------------------------------------------

    def rewrite_query(self, query: str, history: list[dict] | None = None) -> str:
        """Rewrite *query* as a standalone question when history is present.

        Returns the original query unchanged when there is no history to resolve.
        """
        return self._memory.rewrite_query(query, history) if history else query

    def retrieve(
        self,
        query: str,
        rerank: bool = False,
        hybrid: bool = False,
        top_k: int = TOP_K,
        max_chunks_per_source: int = MAX_CHUNKS_PER_SOURCE,
    ) -> list:
        """Embed *query* and return the top-k most relevant Document objects."""
        results = self._retriever.retrieve(
            query,
            top_k=top_k,
            rerank=rerank,
            hybrid=hybrid,
            max_chunks_per_source=max_chunks_per_source,
        )
        return [doc for doc, _ in results]

    def generate(
        self,
        query: str,
        docs: list,
        history: list[dict] | None = None,
        rewritten_query: str = "",
    ) -> Answer:
        """Call the LLM with *docs* as context and return a parsed Answer.

        Returns a no-citation fallback Answer when *docs* is empty.
        """
        if not docs:
            return Answer(text="I don't have enough information to answer that question.")
        raw = self._get_generator().generate(query, docs, history)
        return Answer.from_raw(raw, rewritten_query=rewritten_query)

    def stream_generate(
        self,
        query: str,
        docs: list,
        history: list[dict] | None = None,
    ) -> Iterator[str]:
        """Yield LLM response tokens one at a time.

        Yields a single fallback string immediately when *docs* is empty.
        """
        if not docs:
            yield "I don't have enough information to answer that question."
            return
        yield from self._get_generator().stream(query, docs, history)

    # ------------------------------------------------------------------
    # High-level convenience (used by tests and non-UI callers)
    # ------------------------------------------------------------------

    def ask(
        self,
        query: str,
        history: list[dict] | None = None,
        rerank: bool = False,
        hybrid: bool = False,
    ) -> Answer:
        """Rewrite → retrieve → generate in a single call.

        The original query + full history is sent to the generator so the LLM
        answers in conversational context even when the retrieval query was rewritten.
        """
        rewritten = self.rewrite_query(query, history)
        docs = self.retrieve(rewritten, rerank=rerank, hybrid=hybrid)
        rewritten_for_display = rewritten if rewritten != query else ""
        return self.generate(query, docs, history, rewritten_query=rewritten_for_display)

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
