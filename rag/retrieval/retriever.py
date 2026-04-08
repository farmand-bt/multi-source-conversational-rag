from config.settings import TOP_K
from rag.embeddings.embedder import Embedder
from rag.ingestion.base import Document
from rag.vectorstore.chroma_store import ChromaStore


class Retriever:
    """Embeds a query and retrieves the most relevant chunks from ChromaStore."""

    def __init__(self, embedder: Embedder, store: ChromaStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[tuple[Document, float]]:
        """Return (Document, similarity_score) pairs for the given query.

        Returns an empty list if the query is blank or the store is empty.
        """
        if not query.strip():
            return []
        embedding = self._embedder.embed_query(query)
        return self._store.query(embedding, top_k)
