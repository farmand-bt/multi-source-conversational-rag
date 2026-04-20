from __future__ import annotations

from config.settings import MAX_CHUNKS_PER_SOURCE, RERANK_MODEL, TOP_K
from rag.embeddings.embedder import Embedder
from rag.ingestion.base import Document
from rag.retrieval.bm25_index import BM25Index, reciprocal_rank_fusion
from rag.vectorstore.chroma_store import ChromaStore


class Retriever:
    """Embeds a query and retrieves the most relevant chunks from ChromaStore.

    Applies a per-source cap (MAX_CHUNKS_PER_SOURCE) so that a single
    high-scoring source cannot crowd out all other ingested sources.

    Optionally fuses BM25 keyword results with dense results via Reciprocal
    Rank Fusion (hybrid=True) before the per-source cap.

    Optionally re-ranks candidates with a cross-encoder after fusion.
    The cross-encoder model is loaded lazily on first use (no startup cost).
    """

    def __init__(
        self,
        embedder: Embedder,
        store: ChromaStore,
        bm25_index: BM25Index | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._bm25_index = bm25_index
        self._cross_encoder = None  # lazy-loaded on first rerank call

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        rerank: bool = False,
        hybrid: bool = False,
        max_chunks_per_source: int = MAX_CHUNKS_PER_SOURCE,
    ) -> list[tuple[Document, float]]:
        """Return up to top_k (Document, score) pairs, ordered by score.

        At most max_chunks_per_source chunks are returned from any single source,
        ensuring multiple ingested sources can contribute to the context.
        Returns an empty list if the query is blank or the store is empty.

        Args:
            query:                 The (possibly rewritten) search query.
            top_k:                 Maximum number of results to return.
            rerank:                If True, re-score the candidate pool with a cross-encoder
                                   before applying the per-source cap. More accurate but slower.
            hybrid:                If True, fuse dense vector results with BM25 keyword results
                                   via Reciprocal Rank Fusion before re-ranking/capping.
            max_chunks_per_source: Cap on chunks from any single source (diversity control).
        """
        if not query.strip():
            return []
        embedding = self._embedder.embed_query(query)
        # Fetch a larger pool so the per-source cap still yields top_k results
        candidates = self._store.query(embedding, top_k * max_chunks_per_source)

        if hybrid and self._bm25_index is not None and not self._bm25_index.is_empty:
            bm25_results = self._bm25_index.query(query, top_k * max_chunks_per_source)
            candidates = reciprocal_rank_fusion(candidates, bm25_results)

        if rerank and candidates:
            candidates = self._rerank(query, candidates)

        seen: dict[str, int] = {}
        result: list[tuple[Document, float]] = []
        for doc, score in candidates:
            count = seen.get(doc.source_id, 0)
            if count < max_chunks_per_source:
                result.append((doc, score))
                seen[doc.source_id] = count + 1
            if len(result) >= top_k:
                break
        return result

    def _rerank(
        self, query: str, candidates: list[tuple[Document, float]]
    ) -> list[tuple[Document, float]]:
        """Re-score candidates with a cross-encoder and return them sorted by new score."""
        from sentence_transformers import CrossEncoder

        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder(RERANK_MODEL)

        pairs = [(query, doc.text) for doc, _ in candidates]
        scores: list[float] = self._cross_encoder.predict(pairs).tolist()
        rescored = [(doc, score) for (doc, _), score in zip(candidates, scores)]
        return sorted(rescored, key=lambda x: x[1], reverse=True)
