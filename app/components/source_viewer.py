from datetime import datetime
from datetime import timezone as dt_tz
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import streamlit as st

from rag.pipeline import RAGPipeline

_ICONS = {"pdf": "📄", "web": "🌐", "youtube": "▶️", "text": "📝"}


def _format_ingested(utc_iso: str) -> str:
    """Convert a UTC ISO timestamp to the user's local time (or UTC as fallback)."""
    tz_name: str | None = st.session_state.get("user_tz")
    try:
        dt = datetime.fromisoformat(utc_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=dt_tz.utc)
        if tz_name:
            dt = dt.astimezone(ZoneInfo(tz_name))
            return dt.strftime("%Y-%m-%d %H:%M")
        return dt.strftime("%Y-%m-%d %H:%M") + " UTC"
    except (ValueError, ZoneInfoNotFoundError):
        return utc_iso[:19].replace("T", " ") + " UTC"


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
                st.write(f"**Ingested:** {_format_ingested(source['ingested_at'])}")

            if st.button("Delete", key=f"del_{source['source_id']}", type="secondary"):
                pipeline.delete_source(source["source_id"])
                st.success(f"Deleted {source['source_name']}")
                st.rerun()
