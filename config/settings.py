import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _get(key: str, default: str = "") -> str:
    """Read a setting from environment variables, then Streamlit secrets as fallback.

    Local dev:          values come from .env via python-dotenv.
    Streamlit Cloud:    values come from st.secrets (set in the dashboard).
    """
    value = os.getenv(key, "")
    if not value:
        try:
            import streamlit as st  # noqa: PLC0415

            value = st.secrets.get(key, default)
        except Exception:
            value = default
    return value


# LLM
GWDG_API_KEY = _get("GWDG_API_KEY")
GWDG_API_BASE = _get("GWDG_API_BASE")
GWDG_MODEL_NAME = _get("GWDG_MODEL_NAME")

# Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval
TOP_K = 5
MAX_CHUNKS_PER_SOURCE = 3  # max chunks retrieved from any single source (ensures diversity)
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # local cross-encoder, no API needed

# Memory
MAX_HISTORY_TURNS = 10

# Public app guardrail — max LLM questions per browser session (0 = unlimited)
MAX_QUESTIONS_PER_SESSION = 10

# Vector store — absolute path so it resolves correctly regardless of cwd
CHROMA_PERSIST_DIR = str(Path(__file__).parent.parent / "data" / "chroma")
