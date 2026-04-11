from urllib.parse import urlparse

import streamlit as st

from rag.pipeline import RAGPipeline

_ICONS = {"pdf": "📄", "web": "🌐", "youtube": "▶️"}


def render_source_viewer(pipeline: RAGPipeline) -> None:
    sources = pipeline.list_sources()

    # ── Header row with "Delete all" (only when sources exist) ────────
    col_title, col_btn = st.columns([3, 2])
    with col_title:
        st.header("Ingested Sources")
    with col_btn:
        if sources:
            st.write("")  # vertical alignment nudge
            if st.session_state.get("confirm_delete_all"):
                if st.button(
                    "⚠️ Confirm delete all",
                    type="primary",
                    use_container_width=True,
                ):
                    pipeline.delete_all_sources()
                    st.session_state.confirm_delete_all = False
                    st.rerun()
            else:
                if st.button("🗑️ Delete all", use_container_width=True):
                    st.session_state.confirm_delete_all = True
                    st.rerun()

    if not sources:
        st.session_state.confirm_delete_all = False
        st.caption("No sources yet. Upload a PDF or add a URL from the sidebar.")
        return

    # ── Per-source expanders ───────────────────────────────────────────
    for source in sources:
        icon = _ICONS.get(source["source_type"], "📁")
        label = f"{icon} {source['source_name']}  ·  {source['chunk_count']} chunks"

        with st.expander(label):
            st.write(f"**Type:** {source['source_type'].upper()}")
            st.write(f"**Chunks:** {source['chunk_count']}")

            stype = source["source_type"]
            if stype == "pdf" and source.get("page_count"):
                st.write(f"**Pages:** {source['page_count']}")
            elif stype == "web" and source.get("url"):
                domain = urlparse(source["url"]).netloc
                st.write(f"**Domain:** {domain}")
                st.write(f"**URL:** {source['url']}")
            elif stype == "youtube" and source.get("url"):
                st.write(f"**Video:** {source['url']}")

            if source.get("ingested_at"):
                st.write(f"**Ingested:** {source['ingested_at'][:19].replace('T', ' ')} UTC")

            if st.button("Delete", key=f"del_{source['source_id']}", type="secondary"):
                pipeline.delete_source(source["source_id"])
                st.success(f"Deleted {source['source_name']}")
                st.rerun()
