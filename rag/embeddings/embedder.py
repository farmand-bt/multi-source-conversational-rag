from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL


class Embedder:
    """Thin wrapper around sentence-transformers. Model is loaded once on init."""

    def __init__(self) -> None:
        self._model = SentenceTransformer(EMBEDDING_MODEL)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # normalize_embeddings=True required for correct cosine similarity in ChromaDB
        return self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True).tolist()
