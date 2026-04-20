"""Tests for BM25Index, reciprocal_rank_fusion, and hybrid retrieval.

BM25Index and RRF unit tests run fully offline (no embedder, no network).
Hybrid retrieval integration tests use the session-scoped embedder fixture
from conftest.py.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from rag.ingestion.base import Document
from rag.retrieval.bm25_index import BM25Index, reciprocal_rank_fusion
from rag.retrieval.retriever import Retriever
from rag.vectorstore.chroma_store import ChromaStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(
    text: str,
    source_id: str = "src1",
    chunk_index: int = 0,
    source_name: str = "test.pdf",
) -> Document:
    return Document(
        text=text,
        source_type="pdf",
        source_name=source_name,
        source_id=source_id,
        chunk_index=chunk_index,
    )


# ---------------------------------------------------------------------------
# BM25Index — unit tests (no embedder)
# ---------------------------------------------------------------------------


class TestBM25IndexEmpty:
    def test_empty_index_returns_empty(self):
        assert BM25Index().query("anything", 5) == []

    def test_is_empty_on_new_instance(self):
        assert BM25Index().is_empty

    def test_add_empty_list_is_noop(self):
        idx = BM25Index()
        idx.add([])
        assert idx.is_empty

    def test_query_blank_string_returns_empty(self):
        idx = BM25Index()
        idx.add([_doc("Python is a programming language.")])
        assert idx.query("", 5) == []
        assert idx.query("   ", 5) == []


class TestBM25IndexAdd:
    def test_not_empty_after_add(self):
        idx = BM25Index()
        idx.add([_doc("Python is a programming language.")])
        assert not idx.is_empty

    def test_query_returns_matching_doc(self):
        idx = BM25Index()
        idx.add([_doc("The recipe calls for two eggs and flour.")])
        results = idx.query("eggs", 5)
        assert len(results) == 1
        assert "eggs" in results[0][0].text

    def test_keyword_match_ranks_first(self):
        idx = BM25Index()
        idx.add(
            [
                _doc("The Eiffel Tower is in Paris.", source_id="s1", chunk_index=0),
                _doc("Python is widely used for machine learning.", source_id="s2", chunk_index=0),
                _doc("Python is a high-level programming language.", source_id="s3", chunk_index=0),
            ]
        )
        results = idx.query("Python programming", 3)
        texts = [doc.text for doc, _ in results]
        # At least one Python-related doc should rank in the top 2
        assert any("Python" in t for t in texts[:2])

    def test_scores_ordered_descending(self):
        idx = BM25Index()
        idx.add(
            [
                _doc("Python is used for data science.", source_id="s1", chunk_index=0),
                _doc("The weather is sunny today.", source_id="s2", chunk_index=0),
            ]
        )
        results = idx.query("Python data science", 5)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_matching_doc_ranks_above_non_matching(self):
        # BM25Plus gives every document a positive floor score, so non-matching docs
        # are not strictly excluded — but the matching doc must rank higher.
        idx = BM25Index()
        idx.add(
            [
                _doc("Python is a programming language.", source_id="s1", chunk_index=0),
                _doc("xyzzy foobar qux quux.", source_id="s2", chunk_index=0),
            ]
        )
        results = idx.query("Python", 5)
        assert len(results) >= 1
        assert "Python" in results[0][0].text


class TestBM25IndexDelete:
    def test_delete_removes_source(self):
        idx = BM25Index()
        idx.add([_doc("Python doc", source_id="keep", chunk_index=0)])
        idx.add([_doc("Ruby doc", source_id="remove", chunk_index=0)])
        idx.delete("remove")
        results = idx.query("Ruby", 5)
        source_ids = [doc.source_id for doc, _ in results]
        assert "remove" not in source_ids

    def test_delete_nonexistent_source_is_safe(self):
        idx = BM25Index()
        idx.add([_doc("some text")])
        idx.delete("nonexistent")  # must not raise
        assert not idx.is_empty

    def test_delete_all_clears_index(self):
        idx = BM25Index()
        idx.add([_doc("Python is great.", source_id="s1", chunk_index=0)])
        idx.add([_doc("Ruby is nice.", source_id="s2", chunk_index=0)])
        idx.delete_all()
        assert idx.is_empty
        assert idx.query("Python", 5) == []

    def test_delete_all_then_add_works(self):
        idx = BM25Index()
        idx.add([_doc("first batch")])
        idx.delete_all()
        idx.add([_doc("second batch")])
        results = idx.query("second", 5)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion — unit tests (no embedder)
# ---------------------------------------------------------------------------


class TestReciprocalRankFusion:
    def test_empty_both_returns_empty(self):
        assert reciprocal_rank_fusion([], []) == []

    def test_empty_sparse_returns_dense_order(self):
        d1 = _doc("alpha", source_id="s1", chunk_index=0)
        d2 = _doc("beta", source_id="s2", chunk_index=0)
        result = reciprocal_rank_fusion([(d1, 0.9), (d2, 0.7)], [])
        doc_ids = [(doc.source_id, doc.chunk_index) for doc, _ in result]
        assert doc_ids == [("s1", 0), ("s2", 0)]

    def test_empty_dense_returns_sparse_order(self):
        d1 = _doc("alpha", source_id="s1", chunk_index=0)
        d2 = _doc("beta", source_id="s2", chunk_index=0)
        result = reciprocal_rank_fusion([], [(d1, 10.0), (d2, 5.0)])
        doc_ids = [(doc.source_id, doc.chunk_index) for doc, _ in result]
        assert doc_ids == [("s1", 0), ("s2", 0)]

    def test_shared_doc_receives_higher_score_than_exclusive(self):
        shared = _doc("shared", source_id="shared", chunk_index=0)
        exclusive = _doc("exclusive", source_id="excl", chunk_index=0)
        # shared appears #1 in dense, exclusive appears #1 in sparse
        result = reciprocal_rank_fusion(
            [(shared, 0.9)],
            [(shared, 8.0), (exclusive, 5.0)],
        )
        scores = {doc.source_id: score for doc, score in result}
        assert scores["shared"] > scores["excl"]

    def test_deduplicates_same_doc(self):
        doc = _doc("text", source_id="s1", chunk_index=0)
        result = reciprocal_rank_fusion([(doc, 0.9)], [(doc, 5.0)])
        # Only one entry for the same (source_id, chunk_index)
        keys = [(d.source_id, d.chunk_index) for d, _ in result]
        assert len(keys) == len(set(keys))

    def test_deduplicates_by_source_id_and_chunk_index(self):
        # Same source_id but different chunk_index = different docs
        d0 = _doc("chunk 0", source_id="src", chunk_index=0)
        d1 = _doc("chunk 1", source_id="src", chunk_index=1)
        result = reciprocal_rank_fusion([(d0, 0.9)], [(d1, 5.0)])
        assert len(result) == 2

    def test_union_of_all_docs_returned(self):
        d1 = _doc("a", source_id="s1", chunk_index=0)
        d2 = _doc("b", source_id="s2", chunk_index=0)
        d3 = _doc("c", source_id="s3", chunk_index=0)
        result = reciprocal_rank_fusion([(d1, 0.9), (d2, 0.7)], [(d2, 5.0), (d3, 3.0)])
        source_ids = {doc.source_id for doc, _ in result}
        assert source_ids == {"s1", "s2", "s3"}

    def test_scores_sorted_descending(self):
        d1 = _doc("a", source_id="s1", chunk_index=0)
        d2 = _doc("b", source_id="s2", chunk_index=0)
        d3 = _doc("c", source_id="s3", chunk_index=0)
        result = reciprocal_rank_fusion(
            [(d1, 0.9), (d2, 0.7)],
            [(d2, 5.0), (d3, 3.0)],
        )
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Hybrid retrieval integration tests (use embedder fixture)
# ---------------------------------------------------------------------------


@pytest.fixture
def hybrid_store(tmp_path):
    return ChromaStore(persist_dir=str(tmp_path / "chroma_hybrid"))


@pytest.fixture
def hybrid_retriever(embedder, hybrid_store):
    bm25 = BM25Index()
    return Retriever(embedder=embedder, store=hybrid_store, bm25_index=bm25), bm25


def _seed_hybrid(embedder, store, bm25, docs):
    embs = embedder.embed_documents([d.text for d in docs])
    store.add(docs, embs)
    bm25.add(docs)


class TestHybridRetrieval:
    def test_hybrid_retrieve_returns_results(self, embedder, hybrid_store, hybrid_retriever):
        retriever, bm25 = hybrid_retriever
        docs = [_doc("Python is a high-level programming language.", source_id="s1", chunk_index=0)]
        _seed_hybrid(embedder, hybrid_store, bm25, docs)
        results = retriever.retrieve("Python", top_k=5, hybrid=True)
        assert len(results) >= 1

    def test_hybrid_false_does_not_use_bm25(self, embedder, hybrid_store):
        # With hybrid=False the BM25 index is never consulted even if it contains data
        bm25 = BM25Index()
        retriever = Retriever(embedder=embedder, store=hybrid_store, bm25_index=bm25)
        docs = [_doc("Python programming language", source_id="s1", chunk_index=0)]
        _seed_hybrid(embedder, hybrid_store, bm25, docs)
        # Should work fine — just uses dense path
        results = retriever.retrieve("Python", top_k=5, hybrid=False)
        assert isinstance(results, list)

    def test_hybrid_with_no_bm25_index_falls_back_to_dense(self, embedder, tmp_path):
        store = ChromaStore(persist_dir=str(tmp_path / "chroma_no_bm25"))
        retriever = Retriever(embedder=embedder, store=store, bm25_index=None)
        doc = _doc("Python is great.", source_id="s1", chunk_index=0)
        embs = embedder.embed_documents([doc.text])
        store.add([doc], embs)
        # hybrid=True with bm25_index=None must not raise, falls back to dense
        results = retriever.retrieve("Python", top_k=5, hybrid=True)
        assert len(results) == 1

    def test_hybrid_compatible_with_rerank(
        self, embedder, hybrid_store, hybrid_retriever, monkeypatch
    ):
        retriever, bm25 = hybrid_retriever
        docs = [
            _doc("Deep learning uses neural networks.", source_id="s1", chunk_index=0),
            _doc("Python is a programming language.", source_id="s2", chunk_index=0),
        ]
        _seed_hybrid(embedder, hybrid_store, bm25, docs)

        mock_ce = MagicMock()
        mock_ce.predict.return_value = np.array([0.9, 0.1])
        mock_ce_cls = MagicMock(return_value=mock_ce)
        import sentence_transformers

        monkeypatch.setattr(sentence_transformers, "CrossEncoder", mock_ce_cls)
        retriever._cross_encoder = None

        results = retriever.retrieve("deep learning", top_k=2, hybrid=True, rerank=True)
        assert len(results) >= 1

    def test_per_source_cap_applies_after_hybrid(self, embedder, tmp_path):
        from config.settings import MAX_CHUNKS_PER_SOURCE

        store = ChromaStore(persist_dir=str(tmp_path / "chroma_cap"))
        bm25 = BM25Index()
        retriever = Retriever(embedder=embedder, store=store, bm25_index=bm25)

        source_a = [
            _doc(f"Python language concept {i}.", source_id="src_a", chunk_index=i)
            for i in range(4)
        ]
        source_b = [_doc("Ruby is also a programming language.", source_id="src_b", chunk_index=0)]
        all_docs = source_a + source_b
        _seed_hybrid(embedder, store, bm25, all_docs)

        results = retriever.retrieve("Python programming language", top_k=5, hybrid=True)
        source_ids = [doc.source_id for doc, _ in results]
        assert source_ids.count("src_a") <= MAX_CHUNKS_PER_SOURCE
