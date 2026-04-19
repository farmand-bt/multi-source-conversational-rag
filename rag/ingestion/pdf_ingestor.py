import hashlib
from pathlib import Path

import fitz  # PyMuPDF

from rag.ingestion.base import Document, Ingestor


class PDFIngestor(Ingestor):
    """Extracts text from PDFs using PyMuPDF — one Document per non-empty page."""

    def ingest(self, source: str | bytes, source_name: str | None = None) -> list[Document]:
        """
        Args:
            source: File path (str) or raw bytes (e.g. from Streamlit file_uploader).
            source_name: Human-readable name; inferred from path if not given.
        """
        if isinstance(source, bytes):
            name = source_name or "document.pdf"
            source_id = hashlib.sha256(source).hexdigest()[:16]
            pdf_context = fitz.open(stream=source, filetype="pdf")
        else:
            name = source_name or Path(source).name
            source_id = hashlib.sha256(Path(source).read_bytes()).hexdigest()[:16]
            pdf_context = fitz.open(source)

        documents = []
        with pdf_context as pdf:
            for page_idx in range(len(pdf)):
                text = pdf[page_idx].get_text().strip()
                if not text:
                    continue
                documents.append(
                    Document(
                        text=text,
                        source_type="pdf",
                        source_name=name,
                        source_id=source_id,
                        chunk_index=0,  # re-assigned by Chunker
                        page_number=page_idx + 1,
                    )
                )

        if not documents:
            raise ValueError(
                f"No text could be extracted from '{name}'. "
                "The PDF may be empty, scanned, or password-protected."
            )
        return documents
