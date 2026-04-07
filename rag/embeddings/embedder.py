# Milestone 2: Wrap sentence-transformers all-MiniLM-L6-v2 (384-dim, CPU-friendly).
# Expose embed_documents(texts: list[str]) -> list[list[float]].


class Embedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("Embedder implemented in Milestone 2")
