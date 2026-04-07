from rag.ingestion.base import Document, Ingestor

# Milestone 2: Implement PDF extraction using PyMuPDF (fitz).
# One Document per page; set source_type="pdf", page_number=<page>.


class PDFIngestor(Ingestor):
    def ingest(self, source: str) -> list[Document]:
        raise NotImplementedError("PDF ingestion implemented in Milestone 2")
