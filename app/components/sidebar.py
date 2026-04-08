import streamlit as st

from rag.pipeline import RAGPipeline


def render_sidebar(pipeline: RAGPipeline) -> None:
    with st.sidebar:
        st.title("Sources")

        # ── PDF ───────────────────────────────────────────────────────
        st.subheader("Upload PDF")
        uploaded = st.file_uploader("Choose a PDF file", type="pdf", label_visibility="collapsed")

        if uploaded is not None:
            st.caption(f"Selected: **{uploaded.name}**")
            if st.button("Ingest PDF", type="primary", use_container_width=True):
                with st.spinner(f"Processing {uploaded.name}…"):
                    try:
                        n = pipeline.ingest(
                            uploaded.getvalue(),
                            source_type="pdf",
                            source_name=uploaded.name,
                        )
                        st.success(f"Stored {n} chunks")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Ingestion failed: {exc}")

        st.divider()

        # ── Web URL (M4) ───────────────────────────────────────────────
        st.subheader("Web URL")
        st.text_input(
            "URL",
            placeholder="https://example.com",
            disabled=True,
            label_visibility="collapsed",
        )

        # ── YouTube (M4) ───────────────────────────────────────────────
        st.subheader("YouTube")
        st.text_input(
            "YouTube URL",
            placeholder="https://youtube.com/watch?v=…",
            disabled=True,
            label_visibility="collapsed",
        )

        st.caption("Web & YouTube ingestion coming in Milestone 4")
