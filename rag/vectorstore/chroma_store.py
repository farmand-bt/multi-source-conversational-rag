import chromadb

from config.settings import CHROMA_PERSIST_DIR
from rag.ingestion.base import Document


class ChromaStore:
    """ChromaDB wrapper. Embeddings are computed externally and passed in explicitly."""

    def __init__(
        self,
        persist_dir: str | None = None,
        ephemeral: bool = False,
        collection_name: str = "documents",
    ) -> None:
        # Each ephemeral session must use a unique collection name.
        # ChromaDB 1.x with the Rust backend shares in-process memory between
        # EphemeralClient() instances, so all clients that use the same collection
        # name "documents" would see each other's data.  A per-session UUID name
        # keeps collections fully isolated without requiring separate processes.
        self._collection_name = collection_name
        if ephemeral:
            self._client = chromadb.EphemeralClient()
        else:
            path = persist_dir or CHROMA_PERSIST_DIR
            self._client = chromadb.PersistentClient(path=path)
        # cosine similarity is standard for sentence-transformer embeddings
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, chunks: list[Document], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[f"{c.source_id}_{c.chunk_index}" for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[self._to_metadata(c) for c in chunks],
            embeddings=embeddings,
        )

    def delete(self, source_id: str) -> None:
        self._collection.delete(where={"source_id": source_id})

    def delete_all(self) -> None:
        """Drop and recreate the collection, removing every stored chunk."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(self, embedding: list[float], top_k: int) -> list[tuple[Document, float]]:
        """Return top-k (Document, similarity_score) pairs, ordered by relevance."""
        count = self._collection.count()
        if count == 0:
            return []
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        docs = results.get("documents") or []
        metas = results.get("metadatas") or []
        dists = results.get("distances") or []
        if not docs or not docs[0]:
            return []
        return [
            (self._from_metadata(text, meta), round(1.0 - dist, 4))
            for text, meta, dist in zip(docs[0], metas[0], dists[0])
        ]

    def list_sources(self) -> list[dict]:
        results = self._collection.get(include=["metadatas"])
        sources: dict[str, dict] = {}
        for meta in results.get("metadatas") or []:
            sid = meta["source_id"]
            if sid not in sources:
                sources[sid] = {
                    "source_id": sid,
                    "source_name": meta.get("source_name", "unknown"),
                    "source_type": meta.get("source_type", "unknown"),
                    "ingested_at": meta.get("ingested_at", ""),
                    "chunk_count": 0,
                    "url": meta.get("url"),  # web / youtube
                    "page_count": 0,  # pdf: max page_number seen
                }
            sources[sid]["chunk_count"] += 1
            page = meta.get("page_number") or 0
            if page > sources[sid]["page_count"]:
                sources[sid]["page_count"] = page
        return list(sources.values())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_metadata(doc: Document) -> dict:
        """Convert Document to a flat metadata dict (no None values — ChromaDB rejects them)."""
        meta: dict = {
            "source_type": doc.source_type,
            "source_name": doc.source_name,
            "source_id": doc.source_id,
            "chunk_index": doc.chunk_index,
            "ingested_at": doc.ingested_at,
        }
        if doc.page_number is not None:
            meta["page_number"] = doc.page_number
        if doc.url is not None:
            meta["url"] = doc.url
        if doc.timestamp is not None:
            meta["timestamp"] = doc.timestamp
        return meta

    @staticmethod
    def _from_metadata(text: str, meta: dict) -> Document:
        return Document(
            text=text,
            source_type=meta["source_type"],
            source_name=meta["source_name"],
            source_id=meta["source_id"],
            chunk_index=meta["chunk_index"],
            ingested_at=meta.get("ingested_at", ""),
            page_number=meta.get("page_number"),
            url=meta.get("url"),
            timestamp=meta.get("timestamp"),
        )
