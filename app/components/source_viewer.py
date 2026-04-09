from urllib.parse import urlparse

import streamlit as st

from rag.pipeline import RAGPipeline

_ICONS = {"pdf": "📄", "web": "🌐", "youtube": "▶️"}


def render_source_viewer(pipeline: RAGPipeline) -> None:
    st.header("Ingested Sources")
    sources = pipeline.list_sources()

    if not sources:
        st.caption("No sources yet. Upload a PDF or add a URL from the sidebar.")
        return

    for source in sources:
        icon = _ICONS.get(source["source_type"], "📁")
        label = f"{icon} {source['source_name']}  ·  {source['chunk_count']} chunks"

        with st.expander(label):
            st.write(f"**Type:** {source['source_type'].upper()}")
            st.write(f"**Chunks:** {source['chunk_count']}")

            # Source-type-specific metadata
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
