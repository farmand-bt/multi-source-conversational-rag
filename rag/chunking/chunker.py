from rag.ingestion.base import Document

# Milestone 2: Implement RecursiveCharacterTextSplitter chunking.
# Use CHUNK_SIZE=500, CHUNK_OVERLAP=50 from config.settings.
# Re-assign chunk_index on each split Document; propagate all metadata from the source Document.


class Chunker:
    def chunk(self, documents: list[Document]) -> list[Document]:
        raise NotImplementedError("Chunking implemented in Milestone 2")
