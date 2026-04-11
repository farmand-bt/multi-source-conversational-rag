"""End-to-end pipeline tests (ingest -> query -> answer with citations).

The Generator is replaced with a mock so tests don't require GWDG credentials.
RAGPipeline is constructed via __new__ + manual attribute injection so each test
gets an isolated ChromaStore without touching the real data/ directory.
"""

from rag.chunking.chunker import Chunker
from rag.ingestion.base import Document
from rag.models import Answer
from rag.pipeline import RAGPipeline
from rag.retrieval.retriever import Retriever
from rag.vectorstore.chroma_store import ChromaStore


class _MockGenerator:
    """Returns a canned answer that includes a parseable citation."""

    def generate(self, query: str, context_docs: list, history=None) -> str:
        name = context_docs[0].source_name if context_docs else "unknown"
        page = context_docs[0].page_number if context_docs else 1
        return f"Mock answer. [PDF: {name}, page {page}]"


class _MockMemory:
    """No-op memory: returns query unchanged, ignores state calls."""

    def rewrite_query(self, query: str, history=None) -> str:
        return query

    def add_turn(self, user_message: str, assistant_message: str) -> None:
        pass

    def get_history(self) -> list[dict]:
        return []

    def clear(self) -> None:
        pass


def _build_pipeline(tmp_path, embedder):
    store = ChromaStore(persist_dir=str(tmp_path / "chroma"))
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline._embedder = embedder
    pipeline._chunker = Chunker()
    pipeline._store = store
    pipeline._pdf_ingestor = None  # not needed for read-path tests
    pipeline._retriever = Retriever(embedder, store)
    pipeline._generator = _MockGenerator()
    pipeline._memory = _MockMemory()
    return pipeline


def _seed(pipeline, texts: list[str], source_id: str = "seed001") -> None:
    docs = [
        Document(
            text=text,
            source_type="pdf",
            source_name="seed.pdf",
            source_id=source_id,
            chunk_index=i,
            page_number=i + 1,
        )
        for i, text in enumerate(texts)
    ]
    embs = pipeline._embedder.embed_documents([d.text for d in docs])
    pipeline._store.add(docs, embs)


# ------------------------------------------------------------------
# ask() — read path
# ------------------------------------------------------------------


def test_ask_empty_store_returns_no_info_answer(tmp_path, embedder):
    pipeline = _build_pipeline(tmp_path, embedder)
    answer = pipeline.ask("What is Python?")
    assert isinstance(answer, Answer)
    assert answer.text  # non-empty fallback message
    assert answer.citations == ()


def test_ask_returns_answer_with_citations(tmp_path, embedder):
    pipeline = _build_pipeline(tmp_path, embedder)
    _seed(pipeline, ["Python is a popular high-level programming language."])
    answer = pipeline.ask("What is Python?")
    assert isinstance(answer, Answer)
    assert len(answer.citations) == 1
    assert answer.citations[0].source_type == "PDF"
    assert answer.citations[0].source_name == "seed.pdf"
    assert answer.citations[0].location == "page 1"


def test_ask_passes_history_to_generator(tmp_path, embedder):
    received = {}

    class _HistoryCapture:
        def generate(self, query, context_docs, history=None):
            received["history"] = history
            return "Answer [PDF: seed.pdf, page 1]"

    pipeline = _build_pipeline(tmp_path, embedder)
    pipeline._generator = _HistoryCapture()
    _seed(pipeline, ["Some relevant content."])

    history = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    pipeline.ask("follow-up question", history=history)
    assert received["history"] == history


def test_ask_uses_rewritten_query_for_retrieval(tmp_path, embedder):
    """When history is present, the rewritten query (not the original) is used for retrieval."""
    retrieved_queries = []

    class _CapturingRetriever:
        def retrieve(self, query, top_k=5, rerank=False):
            retrieved_queries.append(query)
            return []

    class _RewritingMemory:
        def rewrite_query(self, query, history=None):
            return "standalone rewritten question"

        def clear(self):
            pass

    pipeline = _build_pipeline(tmp_path, embedder)
    pipeline._retriever = _CapturingRetriever()
    pipeline._memory = _RewritingMemory()

    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    pipeline.ask("follow-up?", history=history)

    assert retrieved_queries == ["standalone rewritten question"]


def test_ask_no_history_skips_rewriting(tmp_path, embedder):
    """Without history the original query goes straight to retrieval."""
    retrieved_queries = []

    class _CapturingRetriever:
        def retrieve(self, query, top_k=5, rerank=False):
            retrieved_queries.append(query)
            return []

    pipeline = _build_pipeline(tmp_path, embedder)
    pipeline._retriever = _CapturingRetriever()

    pipeline.ask("original question")
    assert retrieved_queries == ["original question"]


def test_answer_carries_rewritten_query(tmp_path, embedder):
    """Answer.rewritten_query is populated when rewriting occurs."""

    class _RewritingMemory:
        def rewrite_query(self, query, history=None):
            return "standalone version"

        def clear(self):
            pass

    pipeline = _build_pipeline(tmp_path, embedder)
    pipeline._memory = _RewritingMemory()
    _seed(pipeline, ["Some relevant content."])

    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    answer = pipeline.ask("follow-up?", history=history)
    assert answer.rewritten_query == "standalone version"


def test_clear_history_delegates_to_memory(tmp_path, embedder):
    cleared = []

    class _TrackingMemory:
        def rewrite_query(self, query, history=None):
            return query

        def clear(self):
            cleared.append(True)

    pipeline = _build_pipeline(tmp_path, embedder)
    pipeline._memory = _TrackingMemory()
    pipeline.clear_history()
    assert cleared == [True]


# ------------------------------------------------------------------
# list_sources / delete_source
# ------------------------------------------------------------------


def test_list_sources_after_seed(tmp_path, embedder):
    pipeline = _build_pipeline(tmp_path, embedder)
    _seed(pipeline, ["Content A.", "Content B."])
    sources = pipeline.list_sources()
    assert len(sources) == 1
    assert sources[0]["source_name"] == "seed.pdf"
    assert sources[0]["chunk_count"] == 2


def test_delete_source_removes_chunks(tmp_path, embedder):
    pipeline = _build_pipeline(tmp_path, embedder)
    _seed(pipeline, ["Content to delete."], source_id="del001")
    pipeline.delete_source("del001")
    assert pipeline.list_sources() == []


def test_delete_source_makes_ask_return_fallback(tmp_path, embedder):
    pipeline = _build_pipeline(tmp_path, embedder)
    _seed(pipeline, ["Python is great."], source_id="del002")
    pipeline.delete_source("del002")
    answer = pipeline.ask("What is Python?")
    assert answer.citations == ()
