import re

import streamlit as st

from rag.models import Answer, _CITATION_RE
from rag.pipeline import RAGPipeline

_MAX_HISTORY_TURNS = 10  # user+assistant pairs kept in session state
_CITE_COLOUR = "#e07b39"  # warm orange for inline citation numbers


def render_chat(pipeline: RAGPipeline) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Header row with optional Clear button ─────────────────────────
    col_title, col_btn = st.columns([5, 1])
    with col_title:
        st.header("Chat")
    with col_btn:
        if st.session_state.messages:
            st.write("")  # vertical alignment nudge
            if st.button("Clear", use_container_width=True, help="Clear conversation history"):
                st.session_state.messages = []
                pipeline.clear_history()
                st.rerun()

    # ── Replay all stored messages in order ───────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.markdown(_colorize_refs(msg["content"]), unsafe_allow_html=True)
                if msg.get("rewritten_query"):
                    with st.expander("🔍 Query rewritten as", expanded=False):
                        st.caption(msg["rewritten_query"])
                if msg.get("citations"):
                    _render_sources(msg["citations"])
            else:
                st.write(msg["content"])

    has_sources = bool(pipeline.list_sources())
    placeholder = (
        "Ask a question about your documents…"
        if has_sources
        else "Ingest a source from the sidebar first."
    )

    # ── Handle new input ──────────────────────────────────────────────
    # NOTE: we store everything to session state and call st.rerun() so
    # that all messages replay in the loop above — in the correct order,
    # above the chat input widget.
    if prompt := st.chat_input(placeholder, disabled=not has_sources):
        # Build history from current messages BEFORE appending the new one
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ][-_MAX_HISTORY_TURNS * 2 :]

        st.session_state.messages.append({"role": "user", "content": prompt})

        # Show the question immediately so it's visible while the model thinks
        with st.chat_message("user"):
            st.write(prompt)

        with st.spinner("Thinking…"):
            answer = pipeline.ask(prompt, history=history)

        clean_text, numbered = _number_citations(answer)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": clean_text,
                "citations": numbered,
                "rewritten_query": answer.rewritten_query,
            }
        )

        max_msgs = _MAX_HISTORY_TURNS * 2
        if len(st.session_state.messages) > max_msgs:
            st.session_state.messages = st.session_state.messages[-max_msgs:]

        st.rerun()


# ---------------------------------------------------------------------------
# Citation helpers
# ---------------------------------------------------------------------------

def _number_citations(answer: Answer) -> tuple[str, list[dict]]:
    """Replace [TYPE: name, loc] markers with [N] and group multiple locations.

    Citations from the same source (same type + name) share one number.
    Multiple locations (e.g. two timestamps from one YouTube video) are
    collected into a single entry so the UI can display them together.
    """
    # Group by (source_type, source_name), preserving first-seen order
    groups: dict[tuple[str, str], list[str]] = {}
    order: list[tuple[str, str]] = []

    for c in answer.citations:
        key = (c.source_type, c.source_name)
        if key not in groups:
            groups[key] = []
            order.append(key)
        if c.location and c.location not in groups[key]:
            groups[key].append(c.location)

    num_map: dict[tuple[str, str], int] = {key: i + 1 for i, key in enumerate(order)}

    # Full-key map for text replacement (type + name + location → number)
    full_key_map: dict[tuple[str, str, str], int] = {
        (c.source_type, c.source_name, c.location): num_map[(c.source_type, c.source_name)]
        for c in answer.citations
    }

    def _replace(m: re.Match) -> str:
        key = (m.group(1).strip(), m.group(2).strip(), (m.group(3) or "").strip())
        return f"[{full_key_map.get(key, '?')}]"

    clean_text = _CITATION_RE.sub(_replace, answer.text)
    numbered = [
        {
            "num": num_map[key],
            "source_type": key[0],
            "source_name": key[1],
            "locations": groups[key],  # list; may contain multiple timestamps / pages
        }
        for key in order
    ]
    return clean_text, numbered


def _colorize_refs(text: str) -> str:
    """Wrap [N] citation markers with a coloured bold span."""
    return re.sub(
        r"\[(\d+)\]",
        lambda m: f'<span style="color:{_CITE_COLOUR};font-weight:bold;">[{m.group(1)}]</span>',
        text,
    )


def _render_sources(citations: list[dict]) -> None:
    with st.expander("Sources"):
        for cite in citations:
            st.markdown(_citation_line(cite), unsafe_allow_html=True)


def _citation_line(cite: dict) -> str:
    """Return an HTML/markdown line for one source entry."""
    num = cite["num"]
    src_type = cite["source_type"]
    name = cite["source_name"]
    # Support legacy session state (location: str) and new format (locations: list)
    locations: list[str] = cite.get("locations") or (
        [cite["location"]] if cite.get("location") else []
    )

    badge = f'<span style="color:{_CITE_COLOUR};font-weight:bold;">[{num}]</span>'

    if locations and locations[0].startswith("http"):
        if src_type == "YouTube" and len(locations) > 1:
            # Multiple timestamps from the same video — hyperlink each timestamp
            ts_parts = ", ".join(
                f'[{_ts_label(loc)}]({loc})' for loc in locations
            )
            return f"{badge} {name} — {ts_parts}"
        # Web or single-timestamp YouTube — link the source name
        return f"{badge} [{name}]({locations[0]})"

    # PDF — show name + page numbers (no hyperlink)
    loc_text = ", ".join(locations)
    return f"{badge} {name}, {loc_text}" if loc_text else f"{badge} {name}"


def _ts_label(url: str) -> str:
    """Extract a human-readable MM:SS label from a timestamped YouTube URL (&t=Ns)."""
    m = re.search(r"[&?]t=(\d+)s", url)
    if m:
        total = int(m.group(1))
        mins, secs = divmod(total, 60)
        return f"{mins}:{secs:02d}"
    return "link"
