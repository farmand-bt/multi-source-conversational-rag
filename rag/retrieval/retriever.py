from config.settings import MAX_CHUNKS_PER_SOURCE, TOP_K
from rag.embeddings.embedder import Embedder
from rag.ingestion.base import Document
from rag.vectorstore.chroma_store import ChromaStore


class Retriever:
    """Embeds a query and retrieves the most relevant chunks from ChromaStore.

    Applies a per-source cap (MAX_CHUNKS_PER_SOURCE) so that a single
    high-scoring source cannot crowd out all other ingested sources.
    """

    def __init__(self, embedder: Embedder, store: ChromaStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[tuple[Document, float]]:
        """Return up to top_k (Document, similarity_score) pairs, ordered by score.

        At most MAX_CHUNKS_PER_SOURCE chunks are returned from any single source,
        ensuring multiple ingested sources can contribute to the context.
        Returns an empty list if the query is blank or the store is empty.
        """
        if not query.strip():
            return []
        embedding = self._embedder.embed_query(query)
        # Fetch a larger pool so the per-source cap still yields top_k results
        candidates = self._store.query(embedding, top_k * MAX_CHUNKS_PER_SOURCE)

        seen: dict[str, int] = {}
        result: list[tuple[Document, float]] = []
        for doc, score in candidates:
            count = seen.get(doc.source_id, 0)
            if count < MAX_CHUNKS_PER_SOURCE:
                result.append((doc, score))
                seen[doc.source_id] = count + 1
            if len(result) >= top_k:
                break
        return result
