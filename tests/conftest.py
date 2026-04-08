import pytest

from rag.embeddings.embedder import Embedder


@pytest.fixture(scope="session")
def embedder() -> Embedder:
    """Load the sentence-transformer model once for the entire test session."""
    return Embedder()
