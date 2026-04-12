import streamlit as st
from components.chat import render_chat
from components.sidebar import render_sidebar
from components.source_viewer import render_source_viewer
from page_config import LAYOUT, PAGE_ICON, PAGE_TITLE, SIDEBAR_STATE

from rag.embeddings.embedder import Embedder
from rag.pipeline import RAGPipeline

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state=SIDEBAR_STATE,
)


@st.cache_resource
def _get_shared_embedder() -> Embedder:
    """Load the sentence-transformer model once per server process (shared across all users)."""
    return Embedder()


def _get_pipeline() -> RAGPipeline:
    """Return the per-session pipeline, creating it on first access.

    Each user session gets its own ephemeral (in-memory) ChromaDB so that
    ingested sources are fully isolated between users.  The expensive embedder
    model is shared via _get_shared_embedder().
    """
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGPipeline(
            embedder=_get_shared_embedder(),
            ephemeral=True,
        )
    return st.session_state.pipeline


def main() -> None:
    pipeline = _get_pipeline()

    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.caption("Ask questions across PDFs, web pages, and YouTube transcripts.")

    render_sidebar(pipeline)

    chat_col, sources_col = st.columns([2, 1])
    with chat_col:
        render_chat(pipeline)
    with sources_col:
        render_source_viewer(pipeline)


if __name__ == "__main__":
    main()
