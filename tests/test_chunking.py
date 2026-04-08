import pytest

from rag.chunking.chunker import Chunker
from rag.ingestion.base import Document


def _doc(text: str, **kwargs) -> Document:
    return Document(
        text=text,
        source_type=kwargs.get("source_type", "pdf"),
        source_name=kwargs.get("source_name", "test.pdf"),
        source_id=kwargs.get("source_id", "abc123"),
        chunk_index=0,
        page_number=kwargs.get("page_number", 1),
    )


@pytest.fixture
def chunker() -> Chunker:
    return Chunker()


def test_short_text_produces_single_chunk(chunker):
    chunks = chunker.chunk([_doc("Short text.")])
    assert len(chunks) == 1
    assert chunks[0].text == "Short text."


def test_long_text_is_split(chunker):
    long = "word " * 300  # ~1500 chars — well above chunk_size=500
    chunks = chunker.chunk([_doc(long)])
    assert len(chunks) > 1


def test_chunk_indices_are_sequential(chunker):
    long = "word " * 300
    chunks = chunker.chunk([_doc(long), _doc(long)])
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_metadata_propagated_to_all_chunks(chunker):
    doc = _doc("word " * 300, source_name="report.pdf", page_number=7, source_id="xyz")
    chunks = chunker.chunk([doc])
    for chunk in chunks:
        assert chunk.source_name == "report.pdf"
        assert chunk.page_number == 7
        assert chunk.source_id == "xyz"
        assert chunk.source_type == "pdf"
        assert chunk.ingested_at == doc.ingested_at


def test_multiple_documents_indices_are_globally_sequential(chunker):
    # chunk_index is globally sequential across the entire batch (not per-source),
    # so ChromaDB IDs f"{source_id}_{chunk_index}" remain unique within a single ingest call.
    long = "word " * 300
    docs = [_doc(long, page_number=p) for p in [1, 2, 3]]
    chunks = chunker.chunk(docs)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_empty_document_list(chunker):
    assert chunker.chunk([]) == []
