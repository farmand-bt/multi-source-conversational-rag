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

        # ── Web URL / YouTube (auto-detect) ───────────────────────────
        st.subheader("Add from URL")
        url_input = st.text_input(
            "URL",
            placeholder="https://example.com  or  https://youtube.com/watch?v=…",
            label_visibility="collapsed",
        )

        if url_input:
            is_yt = "youtube.com" in url_input or "youtu.be" in url_input
            source_type = "youtube" if is_yt else "web"
            type_label = "YouTube video" if is_yt else "web page"
            st.caption(f"Detected: {type_label}")

            if st.button(f"Ingest {type_label}", type="primary", use_container_width=True):
                with st.spinner(f"Processing…"):
                    try:
                        n = pipeline.ingest(url_input, source_type=source_type)
                        st.success(f"Stored {n} chunks")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Ingestion failed: {exc}")
