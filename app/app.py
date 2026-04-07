import streamlit as st
from app.config import LAYOUT, PAGE_ICON, PAGE_TITLE, SIDEBAR_STATE

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state=SIDEBAR_STATE,
)


def render_sidebar():
    with st.sidebar:
        st.title("Sources")
        st.caption("Add documents to query against")

        st.subheader("Upload PDF")
        st.file_uploader("Choose a PDF file", type="pdf", disabled=True)

        st.subheader("Web URL")
        st.text_input("Enter a URL", placeholder="https://example.com", disabled=True)

        st.subheader("YouTube")
        st.text_input(
            "Enter a YouTube URL", placeholder="https://youtube.com/watch?v=...", disabled=True
        )

        st.divider()
        st.caption("Ingestion coming in Milestone 2")


def render_chat():
    st.header("Chat")
    st.info("RAG pipeline coming in Milestone 3. Ingest sources from the sidebar first.")

    # Placeholder chat history
    st.chat_message("assistant").write(
        "Hello! Upload a document in the sidebar to get started."
    )

    st.chat_input("Ask a question about your documents...", disabled=True)


def render_source_viewer():
    st.header("Ingested Sources")
    st.caption("Sources added to the knowledge base will appear here.")
    st.empty()


def main():
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.caption("Ask questions across PDFs, web pages, and YouTube transcripts.")

    render_sidebar()

    chat_col, sources_col = st.columns([2, 1])
    with chat_col:
        render_chat()
    with sources_col:
        render_source_viewer()


if __name__ == "__main__":
    main()
