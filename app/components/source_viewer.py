import streamlit as st

from rag.pipeline import RAGPipeline

_ICONS = {"pdf": "📄", "web": "🌐", "youtube": "▶️"}


def render_source_viewer(pipeline: RAGPipeline) -> None:
    st.header("Ingested Sources")
    sources = pipeline.list_sources()

    if not sources:
        st.caption("No sources yet. Upload a PDF from the sidebar.")
        return

    for source in sources:
        icon = _ICONS.get(source["source_type"], "📁")
        label = f"{icon} {source['source_name']}  ·  {source['chunk_count']} chunks"

        with st.expander(label):
            st.write(f"**Type:** {source['source_type'].upper()}")
            st.write(f"**Chunks:** {source['chunk_count']}")
            if source.get("ingested_at"):
                st.write(f"**Ingested:** {source['ingested_at'][:19].replace('T', ' ')} UTC")

            if st.button(
                "Delete", key=f"del_{source['source_id']}", type="secondary"
            ):
                pipeline.delete_source(source["source_id"])
                st.success(f"Deleted {source['source_name']}")
                st.rerun()
