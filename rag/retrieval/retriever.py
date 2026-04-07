from rag.ingestion.base import Document

# Milestone 3: Implement retrieval using ChromaStore + Embedder.
# Optional re-ranking in Milestone 3. Returns top-k Documents.


class Retriever:
    def retrieve(self, query: str) -> list[Document]:
        raise NotImplementedError("Retriever implemented in Milestone 3")
