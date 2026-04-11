from dataclasses import replace

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import CHUNK_OVERLAP, CHUNK_SIZE
from rag.ingestion.base import Document


class Chunker:
    def __init__(self) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks, propagating all metadata.

        chunk_index is assigned sequentially across all input documents.
        All other fields (source_name, source_id, page_number, etc.) are
        copied from the parent Document via dataclasses.replace().
        """
        all_chunks: list[Document] = []
        for doc in documents:
            for text in self._splitter.split_text(doc.text):
                all_chunks.append(replace(doc, text=text, chunk_index=len(all_chunks)))
        return all_chunks
