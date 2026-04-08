import dataclasses

import pytest

from rag.ingestion.base import Document
from rag.retrieval.retriever import Retriever
from rag.vectorstore.chroma_store import ChromaStore


def _doc(text: str, page: int = 1, chunk_index: int = 0) -> Document:
    return Document(
        text=text,
        source_type="pdf",
        source_name="test.pdf",
        source_id="ret_test",
        chunk_index=chunk_index,
        page_number=page,
    )


@pytest.fixture
def store(tmp_path):
    return ChromaStore(persist_dir=str(tmp_path / "chroma"))


@pytest.fixture
def retriever(embedder, store):
    return Retriever(embedder=embedder, store=store)


def test_empty_store_returns_empty(retriever):
    assert retriever.retrieve("What is Python?") == []


def test_blank_query_returns_empty(retriever, embedder, store):
    doc = _doc("Python is a programming language.")
    store.add([doc], embedder.embed_documents([doc.text]))
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_results_are_document_score_tuples(retriever, embedder, store):
    doc = _doc("Machine learning is a branch of AI.")
    store.add([doc], embedder.embed_documents([doc.text]))
    results = retriever.retrieve("machine learning")
    assert len(results) == 1
    retrieved_doc, score = results[0]
    assert isinstance(retrieved_doc, Document)
    assert 0.0 <= score <= 1.0


def test_relevant_docs_rank_higher(retriever, embedder, store):
    docs = [
        _doc("Python is a high-level programming language.", page=1, chunk_index=0),
        _doc("The Eiffel Tower is a famous landmark in Paris.", page=2, chunk_index=1),
        _doc("Python supports object-oriented and functional programming.", page=3, chunk_index=2),
    ]
    embs = embedder.embed_documents([d.text for d in docs])
    store.add(docs, embs)

    results = retriever.retrieve("Python programming language", top_k=2)
    assert len(results) == 2
    top_texts = [doc.text for doc, _ in results]
    assert any("Python" in t for t in top_texts)


def test_top_k_limits_results(retriever, embedder, store):
    docs = [_doc(f"Fact number {i} about nature.", page=i + 1, chunk_index=i) for i in range(6)]
    embs = embedder.embed_documents([d.text for d in docs])
    store.add(docs, embs)
    results = retriever.retrieve("facts about nature", top_k=3)
    assert len(results) <= 3


def test_scores_are_ordered_descending(retriever, embedder, store):
    docs = [
        _doc("Deep learning uses neural networks.", page=1, chunk_index=0),
        _doc("Cooking pasta requires boiling water.", page=2, chunk_index=1),
    ]
    embs = embedder.embed_documents([d.text for d in docs])
    store.add(docs, embs)
    results = retriever.retrieve("deep learning neural networks", top_k=2)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
