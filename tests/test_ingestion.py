import fitz
import pytest

from rag.ingestion.pdf_ingestor import PDFIngestor


def _make_pdf(*page_texts: str) -> bytes:
    """Create an in-memory PDF with one page per text string."""
    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page()
        if text:
            page.insert_text((50, 100), text)
    return doc.tobytes()


@pytest.fixture
def ingestor() -> PDFIngestor:
    return PDFIngestor()


def test_returns_one_document_per_non_empty_page(ingestor):
    pdf = _make_pdf("Page one content", "Page two content")
    docs = ingestor.ingest(pdf, "test.pdf")
    assert len(docs) == 2


def test_skips_empty_pages(ingestor):
    pdf = _make_pdf("", "Has content", "")
    docs = ingestor.ingest(pdf, "test.pdf")
    assert len(docs) == 1
    assert "Has content" in docs[0].text


def test_metadata_source_type(ingestor):
    pdf = _make_pdf("Some text")
    docs = ingestor.ingest(pdf, "my.pdf")
    assert docs[0].source_type == "pdf"


def test_metadata_source_name(ingestor):
    pdf = _make_pdf("Some text")
    docs = ingestor.ingest(pdf, "report.pdf")
    assert docs[0].source_name == "report.pdf"


def test_metadata_page_number(ingestor):
    pdf = _make_pdf("First", "Second", "Third")
    docs = ingestor.ingest(pdf, "test.pdf")
    assert [d.page_number for d in docs] == [1, 2, 3]


def test_source_id_is_stable_hash(ingestor):
    pdf = _make_pdf("Content")
    docs1 = ingestor.ingest(pdf, "a.pdf")
    docs2 = ingestor.ingest(pdf, "a.pdf")
    assert docs1[0].source_id == docs2[0].source_id


def test_source_id_differs_for_different_content(ingestor):
    pdf_a = _make_pdf("Content A")
    pdf_b = _make_pdf("Content B")
    id_a = ingestor.ingest(pdf_a, "a.pdf")[0].source_id
    id_b = ingestor.ingest(pdf_b, "b.pdf")[0].source_id
    assert id_a != id_b


def test_chunk_index_initialized_to_zero(ingestor):
    pdf = _make_pdf("Some text")
    docs = ingestor.ingest(pdf, "test.pdf")
    assert all(d.chunk_index == 0 for d in docs)
