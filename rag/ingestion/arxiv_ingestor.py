import re
import urllib.request
import xml.etree.ElementTree as ET

import requests

from rag.ingestion.base import Document, Ingestor
from rag.ingestion.pdf_ingestor import PDFIngestor

# Matches bare IDs (2305.14283, 2305.14283v2) anywhere in a string / URL
_ID_RE = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?)")


class ArXivIngestor(Ingestor):
    """Download an arXiv paper by ID or URL and ingest it as a PDF.

    The returned Documents have source_type='pdf' because the content is
    extracted identically to an uploaded PDF.  The source_name is set to
    'arXiv:<id>' so citations are clearly distinguishable from local files.
    """

    _PDF_URL = "https://arxiv.org/pdf/{id}"

    def __init__(self) -> None:
        self._pdf_ingestor = PDFIngestor()

    def ingest(self, source: str, source_name: str | None = None) -> list[Document]:
        """
        Args:
            source: arXiv URL (abs or pdf) or bare ID, e.g. '2305.14283' or
                    'https://arxiv.org/abs/2305.14283'.
            source_name: Override the display name; defaults to 'arXiv:<id>'.
        """
        arxiv_id = self._extract_id(source.strip())
        pdf_bytes = self._download(arxiv_id)
        display_name = source_name or _fetch_title(arxiv_id) or f"arXiv:{arxiv_id}"
        return self._pdf_ingestor.ingest(pdf_bytes, display_name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_id(source: str) -> str:
        m = _ID_RE.search(source)
        if m:
            return m.group(1)
        raise ValueError(
            f"Could not parse an arXiv ID from {source!r}. "
            "Provide a URL like https://arxiv.org/abs/2305.14283 "
            "or a bare ID like 2305.14283."
        )

    @staticmethod
    def _download(arxiv_id: str) -> bytes:
        url = ArXivIngestor._PDF_URL.format(id=arxiv_id)
        try:
            resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
        except requests.RequestException as e:
            raise ValueError(f"Failed to download arXiv paper '{arxiv_id}': {e}") from e
        return resp.content


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _fetch_title(arxiv_id: str) -> str | None:
    """Fetch the paper title from the arXiv public API (no key required)."""
    api_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        with urllib.request.urlopen(api_url, timeout=10) as resp:
            xml_data = resp.read().decode()
        root = ET.fromstring(xml_data)
        entry = root.find("atom:entry", _ATOM_NS)
        if entry is not None:
            title_el = entry.find("atom:title", _ATOM_NS)
            if title_el is not None and title_el.text:
                # arXiv titles sometimes contain internal newlines — collapse them
                return " ".join(title_el.text.split())
    except Exception:
        return None
    return None
