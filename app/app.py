import streamlit as st

from components.sidebar import render_sidebar
from components.source_viewer import render_source_viewer
from page_config import LAYOUT, PAGE_ICON, PAGE_TITLE, SIDEBAR_STATE
from rag.pipeline import RAGPipeline

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state=SIDEBAR_STATE,
)


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    """Shared across all sessions; model loaded once per server process."""
    return RAGPipeline()


def render_chat() -> None:
    st.header("Chat")
    st.info("RAG pipeline coming in Milestone 3. Ingest sources from the sidebar first.")
    st.chat_message("assistant").write(
        "Hello! Upload a PDF in the sidebar to get started."
    )
    st.chat_input("Ask a question about your documents…", disabled=True)


def main() -> None:
    pipeline = get_pipeline()

    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.caption("Ask questions across PDFs, web pages, and YouTube transcripts.")

    render_sidebar(pipeline)

    chat_col, sources_col = st.columns([2, 1])
    with chat_col:
        render_chat()
    with sources_col:
        render_source_viewer(pipeline)


if __name__ == "__main__":
    main()
