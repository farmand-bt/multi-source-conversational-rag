"""BM25 keyword index and Reciprocal Rank Fusion for hybrid retrieval.

The BM25Index maintains an in-memory keyword index that mirrors the
ChromaStore. It is kept in sync by the pipeline: add() is called after
every ingest, delete() / delete_all() mirror the corresponding ChromaStore
operations.

reciprocal_rank_fusion() is a pure function that merges a dense result list
and a sparse (BM25) result list into a single ranked list.
"""

import re

from config.settings import RRF_K
from rag.ingestion.base import Document


def _tokenize(text: str) -> list[str]:
    """Lowercase and extract word tokens, stripping punctuation."""
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """In-memory BM25 keyword index over ingested document chunks.

    Maintains three parallel structures that are rebuilt on every mutation:
    - _corpus:     list of tokenized texts (list[list[str]])
    - _documents:  list of Document objects (for returning results)
    - _source_map: source_id -> list of corpus indices (for efficient deletion)

    The rank_bm25 index is rebuilt (O(N)) after each add/delete.  For the
    expected scale — hundreds to low-thousands of chunks per ephemeral session
    — this is negligible.
    """

    def __init__(self) -> None:
        self._corpus: list[list[str]] = []
        self._documents: list[Document] = []
        self._source_map: dict[str, list[int]] = {}
        self._bm25 = None  # BM25Okapi | None — lazy-imported from rank_bm25

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, chunks: list[Document]) -> None:
        if not chunks:
            return
        for doc in chunks:
            idx = len(self._corpus)
            self._corpus.append(_tokenize(doc.text))
            self._documents.append(doc)
            self._source_map.setdefault(doc.source_id, []).append(idx)
        self._rebuild()

    def delete(self, source_id: str) -> None:
        indices_to_remove = set(self._source_map.pop(source_id, []))
        if not indices_to_remove:
            return
        keep = [i for i in range(len(self._corpus)) if i not in indices_to_remove]
        self._corpus = [self._corpus[i] for i in keep]
        self._documents = [self._documents[i] for i in keep]
        # Rebuild source_map with updated indices
        self._source_map = {}
        for new_idx, doc in enumerate(self._documents):
            self._source_map.setdefault(doc.source_id, []).append(new_idx)
        self._rebuild()

    def delete_all(self) -> None:
        self._corpus = []
        self._documents = []
        self._source_map = {}
        self._bm25 = None

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(self, query_text: str, top_k: int) -> list[tuple[Document, float]]:
        """Return up to top_k (Document, bm25_score) pairs, ordered by score.

        Returns an empty list when the index is empty or the query is blank.
        Zero-score documents (no keyword overlap) are excluded.
        """
        if self._bm25 is None or not self._corpus:
            return []
        tokens = _tokenize(query_text)
        if not tokens:
            return []
        scores: list[float] = self._bm25.get_scores(tokens).tolist()
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in ranked[:top_k]:
            if scores[i] > 0:
                results.append((self._documents[i], scores[i]))
        return results

    @property
    def is_empty(self) -> bool:
        return not self._corpus

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _rebuild(self) -> None:
        if self._corpus:
            from rank_bm25 import BM25Plus  # lazy import — not needed until first add()

            # BM25Plus uses idf = log(N+1) - log(df), which is always positive even for
            # single-document corpora (unlike BM25Okapi which can return negative IDF).
            self._bm25 = BM25Plus(self._corpus)
        else:
            self._bm25 = None


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    dense_results: list[tuple[Document, float]],
    sparse_results: list[tuple[Document, float]],
    k: int = RRF_K,
) -> list[tuple[Document, float]]:
    """Merge dense and sparse result lists via Reciprocal Rank Fusion.

    Each document receives a score of 1/(k+rank) from each list it appears in
    (1-based rank).  Scores are summed across lists so documents appearing in
    both receive a bonus.  The merged list is returned sorted by RRF score
    descending.

    Deduplication uses (source_id, chunk_index) as the document key — the same
    unique identifier used by ChromaStore IDs.

    Args:
        dense_results:  Ranked list of (Document, score) from vector search.
        sparse_results: Ranked list of (Document, score) from BM25 search.
        k:              RRF constant (default 60, as in the original paper).

    Returns:
        Merged, deduplicated list sorted by RRF score descending.
    """
    rrf_scores: dict[tuple[str, int], float] = {}
    doc_lookup: dict[tuple[str, int], Document] = {}

    for ranked_list in (dense_results, sparse_results):
        for rank, (doc, _) in enumerate(ranked_list, start=1):
            key = (doc.source_id, doc.chunk_index)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            doc_lookup[key] = doc

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_lookup[key], score) for key, score in merged]
