import streamlit as st

from rag.pipeline import RAGPipeline

# Each counter controls the widget key for the matching URL input.
# Incrementing it on the next rerun forces Streamlit to create a fresh,
# empty widget — the only reliable way to programmatically clear a text_input.
_WEB_KEY = "web_url_counter"
_YT_KEY = "yt_url_counter"
_ARXIV_KEY = "arxiv_id_counter"


def render_sidebar(pipeline: RAGPipeline) -> None:
    # Initialise counters once per session
    for k in (_WEB_KEY, _YT_KEY, _ARXIV_KEY):
        if k not in st.session_state:
            st.session_state[k] = 0

    with st.sidebar:
        st.title("Sources")

        # ── PDF ───────────────────────────────────────────────────────
        st.subheader("📄 Upload PDF")
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

        # ── Web URL ───────────────────────────────────────────────────
        st.subheader("🌐 Web Page")
        web_url = st.text_input(
            "Web URL",
            placeholder="https://example.com/article",
            label_visibility="collapsed",
            key=f"web_url_{st.session_state[_WEB_KEY]}",
        )
        if web_url:
            col1, col2 = st.columns([3, 1])
            with col1:
                ingest_web = st.button(
                    "Ingest", type="primary", use_container_width=True, key="btn_ingest_web"
                )
            with col2:
                if st.button("Clear", use_container_width=True, key="btn_clear_web"):
                    st.session_state[_WEB_KEY] += 1
                    st.rerun()
            if ingest_web:
                with st.spinner("Fetching and extracting…"):
                    try:
                        n = pipeline.ingest(web_url, source_type="web")
                        st.success(f"Stored {n} chunks")
                        st.session_state[_WEB_KEY] += 1
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Ingestion failed: {exc}")

        st.divider()

        # ── YouTube ───────────────────────────────────────────────────
        st.subheader("▶️ YouTube")
        yt_url = st.text_input(
            "YouTube URL",
            placeholder="https://youtube.com/watch?v=…",
            label_visibility="collapsed",
            key=f"yt_url_{st.session_state[_YT_KEY]}",
        )
        if yt_url:
            col1, col2 = st.columns([3, 1])
            with col1:
                ingest_yt = st.button(
                    "Ingest", type="primary", use_container_width=True, key="btn_ingest_yt"
                )
            with col2:
                if st.button("Clear", use_container_width=True, key="btn_clear_yt"):
                    st.session_state[_YT_KEY] += 1
                    st.rerun()
            if ingest_yt:
                with st.spinner("Fetching transcript…"):
                    try:
                        n = pipeline.ingest(yt_url, source_type="youtube")
                        st.success(f"Stored {n} chunks")
                        st.session_state[_YT_KEY] += 1
                        st.rerun()
                    except Exception as exc:
                        err = str(exc).lower()
                        if any(k in err for k in ("too many", "429", "blocked", "ip")):
                            st.error(
                                "YouTube blocked the request — this is common on cloud-hosted "
                                "apps because YouTube restricts server IPs. "
                                "Try the locally-hosted app, or try a different video."
                            )
                        else:
                            st.error(f"Ingestion failed: {exc}")

        st.divider()

        # ── arXiv ─────────────────────────────────────────────────────
        st.subheader("📜 arXiv Paper")
        arxiv_input = st.text_input(
            "arXiv ID or URL",
            placeholder="2305.14283 or https://arxiv.org/abs/2305.14283",
            label_visibility="collapsed",
            key=f"arxiv_id_{st.session_state[_ARXIV_KEY]}",
        )
        if arxiv_input:
            col1, col2 = st.columns([3, 1])
            with col1:
                ingest_arxiv = st.button(
                    "Ingest", type="primary", use_container_width=True, key="btn_ingest_arxiv"
                )
            with col2:
                if st.button("Clear", use_container_width=True, key="btn_clear_arxiv"):
                    st.session_state[_ARXIV_KEY] += 1
                    st.rerun()
            if ingest_arxiv:
                with st.spinner("Downloading PDF from arXiv…"):
                    try:
                        n = pipeline.ingest(arxiv_input, source_type="arxiv")
                        st.success(f"Stored {n} chunks")
                        st.session_state[_ARXIV_KEY] += 1
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Ingestion failed: {exc}")

        st.divider()

        # ── Retrieval settings ────────────────────────────────────────
        st.subheader("⚙️ Retrieval")
        st.toggle(
            "Re-rank results",
            key="use_reranking",
            help=(
                "Uses a cross-encoder to re-score retrieved chunks before generating "
                "an answer. More accurate for follow-up and ambiguous questions, "
                "but adds ~1–3 s latency. Runs locally — no extra API calls."
            ),
        )
