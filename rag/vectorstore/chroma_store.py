from rag.ingestion.base import Document

# Milestone 2: Implement ChromaDB operations persisted at CHROMA_PERSIST_DIR.
# Methods: add(documents), query(embedding, top_k) -> list[Document], delete(source_id).


class ChromaStore:
    def add(self, documents: list[Document]) -> None:
        raise NotImplementedError("ChromaStore implemented in Milestone 2")

    def query(self, embedding: list[float], top_k: int) -> list[Document]:
        raise NotImplementedError("ChromaStore implemented in Milestone 2")

    def delete(self, source_id: str) -> None:
        raise NotImplementedError("ChromaStore implemented in Milestone 2")
