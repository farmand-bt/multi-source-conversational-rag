import re
from datetime import datetime

import streamlit as st

from config.settings import MAX_CHUNKS_PER_SOURCE, MAX_QUESTIONS_PER_SESSION, TOP_K
from rag.models import _CITATION_RE, Answer
from rag.pipeline import RAGPipeline

_MAX_HISTORY_TURNS = 10  # user+assistant pairs kept in session state
_ORANGE = "#e07b39"  # warm orange — used for citations and active pipeline steps
_GREEN = "#198754"
_GREY = "#adb5bd"

# Inject once per page load — justifies assistant answer paragraphs
_JUSTIFY_CSS = """
<style>
[data-testid="stChatMessage"] p {
    text-align: justify;
    hyphens: auto;
}
</style>
"""


def render_chat(pipeline: RAGPipeline) -> None:
    st.markdown(_JUSTIFY_CSS, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "question_count" not in st.session_state:
        st.session_state.question_count = 0

    # ── Header row with Export + Clear buttons ────────────────────────
    col_title, col_export, col_clear = st.columns([4, 1, 1])
    with col_title:
        st.header("Chat")
    if st.session_state.messages:
        with col_export:
            st.write("")  # vertical alignment nudge
            st.download_button(
                "📥 Export",
                data=_export_chat_pdf(st.session_state.messages),
                file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
                help="Download conversation as PDF",
            )
        with col_clear:
            st.write("")
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
    limit_reached = (
        MAX_QUESTIONS_PER_SESSION > 0
        and st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION
    )

    if has_sources and not limit_reached:
        st.caption("💡 Tip: specific questions get better results than broad ones.")

    if limit_reached:
        st.warning(
            f"You've reached the session limit of {MAX_QUESTIONS_PER_SESSION} questions. "
            "Refresh the page to start a new session."
        )
        placeholder = "Session limit reached."
    elif has_sources:
        remaining = MAX_QUESTIONS_PER_SESSION - st.session_state.question_count
        placeholder = (
            f"Ask a question… ({remaining} question{'s' if remaining != 1 else ''} remaining)"
            if MAX_QUESTIONS_PER_SESSION > 0
            else "Ask a question about your documents…"
        )
    else:
        placeholder = "Ingest a source from the sidebar first."

    # ── Handle new input ──────────────────────────────────────────────
    if prompt := st.chat_input(placeholder, disabled=not has_sources or limit_reached):
        # Build history from current messages BEFORE appending the new one.
        # Use raw_content for assistant turns so the LLM sees [PDF: name, page N]
        # citation markers rather than the already-numbered [1], [2] display form.
        history = [
            {"role": m["role"], "content": m.get("raw_content", m["content"])}
            for m in st.session_state.messages
        ][-_MAX_HISTORY_TURNS * 2 :]

        st.session_state.question_count += 1
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Show the question immediately so it's visible while the model thinks
        with st.chat_message("user"):
            st.write(prompt)

        rerank = st.session_state.get("use_reranking", False)
        hybrid = st.session_state.get("use_hybrid", False)
        top_k = st.session_state.get("top_k", TOP_K)
        max_chunks = st.session_state.get("max_chunks_per_source", MAX_CHUNKS_PER_SOURCE)
        answer = _run_pipeline(pipeline, prompt, history, rerank, hybrid, top_k, max_chunks)

        clean_text, numbered = _number_citations(answer)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": clean_text,  # numbered [1],[2] — for display
                "raw_content": answer.text,  # original [PDF: ...] markers — for LLM history
                "citations": numbered,
                "rewritten_query": answer.rewritten_query,
            }
        )

        max_msgs = _MAX_HISTORY_TURNS * 2
        if len(st.session_state.messages) > max_msgs:
            st.session_state.messages = st.session_state.messages[-max_msgs:]

        st.rerun()


# ---------------------------------------------------------------------------
# Pipeline runner — hybrid streaming approach
# ---------------------------------------------------------------------------


def _run_pipeline(
    pipeline: RAGPipeline,
    prompt: str,
    history: list[dict],
    rerank: bool,
    hybrid: bool = False,
    top_k: int = TOP_K,
    max_chunks_per_source: int = MAX_CHUNKS_PER_SOURCE,
) -> Answer:
    """Execute rewrite → retrieve → stream-generate.

    The progress card covers the fast Rewrite and Retrieve steps.
    Once retrieval completes the card is cleared and the LLM response
    streams directly into an assistant chat bubble word-by-word.
    """
    has_history = bool(history)
    if rerank and hybrid:
        retrieve_label = "Hybrid Retrieve & Re-rank"
    elif hybrid:
        retrieve_label = "Hybrid Retrieve"
    elif rerank:
        retrieve_label = "Retrieve & Re-rank"
    else:
        retrieve_label = "Retrieve"

    # Generate is intentionally omitted from the card — the streaming
    # output itself serves as the visual indicator for that step.
    steps: list[dict] = []
    if has_history:
        steps.append({"emoji": "✏️", "label": "Rewrite", "state": "pending", "detail": ""})
    steps.append({"emoji": "🔍", "label": retrieve_label, "state": "pending", "detail": ""})

    box = st.empty()

    def _render() -> None:
        box.markdown(_pipeline_card(steps), unsafe_allow_html=True)

    _render()

    # ── Step: Rewrite ─────────────────────────────────────────────────
    rewritten = prompt
    if has_history:
        steps[0]["state"] = "active"
        _render()
        rewritten = pipeline.rewrite_query(prompt, history)
        steps[0]["state"] = "done"
        if rewritten != prompt:
            short = rewritten if len(rewritten) <= 60 else rewritten[:57] + "…"
            steps[0]["detail"] = f'"{short}"'
        else:
            steps[0]["detail"] = "No change"
        _render()

    # ── Step: Retrieve ────────────────────────────────────────────────
    retrieve_idx = 1 if has_history else 0
    steps[retrieve_idx]["state"] = "active"
    _render()
    docs = pipeline.retrieve(
        rewritten,
        rerank=rerank,
        hybrid=hybrid,
        top_k=top_k,
        max_chunks_per_source=max_chunks_per_source,
    )
    steps[retrieve_idx]["state"] = "done"
    n = len(docs)
    steps[retrieve_idx]["detail"] = f"{n} chunk{'s' if n != 1 else ''} found"
    _render()

    if not docs:
        box.empty()
        return Answer(text="I don't have enough information to answer that question.")

    box.empty()  # progress card cleared — streaming begins immediately

    # ── Step: Generate (streaming) ────────────────────────────────────
    rewritten_for_display = rewritten if rewritten != prompt else ""
    with st.chat_message("assistant"):
        stream_placeholder = st.empty()
        full_text = ""
        try:
            for token in pipeline.stream_generate(prompt, docs, history):
                full_text += token
                stream_placeholder.markdown(full_text + "▌", unsafe_allow_html=True)
        except Exception as exc:
            stream_placeholder.empty()  # clear any partial streamed content
            return Answer(text=f"⚠️ {_llm_error_message(exc)}")

        # Swap raw citation markers for numbered, coloured references
        answer = Answer.from_raw(full_text, rewritten_query=rewritten_for_display)
        clean_text, numbered = _number_citations(answer)
        stream_placeholder.markdown(_colorize_refs(clean_text), unsafe_allow_html=True)
        if answer.rewritten_query:
            with st.expander("🔍 Query rewritten as", expanded=False):
                st.caption(answer.rewritten_query)
        _render_sources(numbered)

    return answer


# ---------------------------------------------------------------------------
# Non-streaming pipeline runner (kept for reference)
# To switch back: replace the _run_pipeline body above with this one.
# ---------------------------------------------------------------------------
#
# def _run_pipeline(
#     pipeline: RAGPipeline,
#     prompt: str,
#     history: list[dict],
#     rerank: bool,
#     top_k: int = TOP_K,
#     max_chunks_per_source: int = MAX_CHUNKS_PER_SOURCE,
# ) -> Answer:
#     """Execute rewrite → retrieve → generate with a live step-by-step progress card.
#
#     Each step updates an st.empty() placeholder immediately, guaranteeing
#     that steps appear sequentially in the UI as they complete.
#     """
#     has_history = bool(history)
#     retrieve_label = "Retrieve & Re-rank" if rerank else "Retrieve"
#
#     # Build the ordered step list (Rewrite only shown when there is history)
#     steps: list[dict] = []
#     if has_history:
#         steps.append({"emoji": "✏️", "label": "Rewrite", "state": "pending", "detail": ""})
#     steps.append({"emoji": "🔍", "label": retrieve_label, "state": "pending", "detail": ""})
#     steps.append({"emoji": "🤖", "label": "Generate", "state": "pending", "detail": ""})
#
#     box = st.empty()
#
#     def _render() -> None:
#         box.markdown(_pipeline_card(steps), unsafe_allow_html=True)
#
#     _render()
#
#     # ── Step: Rewrite ─────────────────────────────────────────────────
#     rewritten = prompt
#     if has_history:
#         steps[0]["state"] = "active"
#         _render()
#         rewritten = pipeline.rewrite_query(prompt, history)
#         steps[0]["state"] = "done"
#         if rewritten != prompt:
#             short = rewritten if len(rewritten) <= 60 else rewritten[:57] + "…"
#             steps[0]["detail"] = f'"{short}"'
#         else:
#             steps[0]["detail"] = "No change"
#         _render()
#
#     # ── Step: Retrieve ────────────────────────────────────────────────
#     retrieve_idx = 1 if has_history else 0
#     steps[retrieve_idx]["state"] = "active"
#     _render()
#     docs = pipeline.retrieve(
#         rewritten, rerank=rerank, top_k=top_k, max_chunks_per_source=max_chunks_per_source
#     )
#     steps[retrieve_idx]["state"] = "done"
#     n = len(docs)
#     steps[retrieve_idx]["detail"] = f"{n} chunk{'s' if n != 1 else ''} found"
#     _render()
#
#     # ── Step: Generate ────────────────────────────────────────────────
#     generate_idx = retrieve_idx + 1
#     if not docs:
#         box.empty()
#         return Answer(text="I don't have enough information to answer that question.")
#
#     steps[generate_idx]["state"] = "active"
#     _render()
#     rewritten_for_display = rewritten if rewritten != prompt else ""
#     try:
#         answer = pipeline.generate(prompt, docs, history, rewritten_query=rewritten_for_display)
#     except Exception as exc:
#         steps[generate_idx]["detail"] = "Error"
#         _render()
#         box.empty()
#         return Answer(text=f"⚠️ {_llm_error_message(exc)}")
#
#     steps[generate_idx]["state"] = "done"
#     _render()
#
#     box.empty()  # remove the progress card once the answer is ready
#     return answer


def _llm_error_message(exc: Exception) -> str:
    """Convert an LLM API exception into a user-friendly message."""
    msg = str(exc).lower()
    if "rate limit" in msg or "429" in msg:
        return "The LLM API rate limit was reached. Please wait a moment and try again."
    if "401" in msg or "unauthorized" in msg or "authentication" in msg:
        return "LLM API authentication failed. Check that your API key is correct."
    if "connect" in msg or "network" in msg or "timeout" in msg:
        return "Could not reach the LLM API. Check your internet connection and try again."
    return f"The LLM returned an unexpected error: {exc}"


def _pipeline_card(steps: list[dict]) -> str:
    """Render a horizontal pipeline progress card as HTML."""
    cells = []
    for i, step in enumerate(steps):
        state = step["state"]

        if state == "pending":
            icon = "○"
            color = _GREY
            sub = "Waiting…"
        elif state == "active":
            icon = "●"
            color = _ORANGE
            sub = "Running…"
        else:  # done
            icon = "✓"
            color = _GREEN
            sub = step.get("detail") or "Done"

        cell = (
            f'<div style="text-align:center;flex:1;min-width:80px;">'
            f'  <div style="font-size:20px;line-height:1.3">{step["emoji"]}</div>'
            f'  <div style="font-size:13px;font-weight:600;color:{color};margin-top:2px">'
            f"    {step['label']}"
            f"  </div>"
            f'  <div style="font-size:11px;color:{color};margin-top:2px">'
            f"    {icon} {sub}"
            f"  </div>"
            f"</div>"
        )
        cells.append(cell)

        if i < len(steps) - 1:
            arrow_col = _GREEN if state == "done" else _GREY
            cells.append(
                f'<div style="padding:18px 4px 0;color:{arrow_col};font-size:16px">→</div>'
            )

    inner = "\n".join(cells)
    return (
        '<div style="display:flex;align-items:flex-start;gap:4px;padding:12px 16px;'
        'background:#f8f9fa;border-radius:8px;border:1px solid #dee2e6;margin:4px 0;">'
        f"{inner}"
        "</div>"
    )


# ---------------------------------------------------------------------------
# PDF export
# ---------------------------------------------------------------------------


def _export_chat_pdf(messages: list[dict]) -> bytes:
    """Render the conversation as a PDF and return the raw bytes."""
    from fpdf import FPDF  # lazy import — only needed when Export is clicked
    from fpdf.enums import XPos, YPos

    def _safe(text: str) -> str:
        """Normalize to latin-1, replacing common Unicode punctuation with ASCII.

        em-dash, smart quotes, etc. are not in ISO-8859-1 and must be mapped
        before encoding, otherwise fpdf2 receives replacement bytes that can
        confuse its line-break algorithm.  Long unbreakable tokens (URLs) are
        split at 55 chars so multi_cell always has a break point.
        """
        text = (
            text.replace("\u2014", "-")
            .replace("\u2013", "-")  # em/en dash
            .replace("\u2018", "'")
            .replace("\u2019", "'")  # smart single quotes
            .replace("\u201c", '"')
            .replace("\u201d", '"')  # smart double quotes
            .replace("\u2026", "...")  # ellipsis
        )
        text = re.sub(r"(\S{55})", r"\1 ", text)
        return text.encode("latin-1", errors="replace").decode("latin-1")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Conversation Export", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(120, 120, 120)
    ts = _safe(datetime.now().strftime("%Y-%m-%d %H:%M"))
    pdf.cell(0, 6, ts, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    for msg in messages:
        is_user = msg["role"] == "user"
        # Role label
        pdf.set_font("Helvetica", "B", 10)
        if is_user:
            pdf.set_text_color(80, 80, 80)
        else:
            pdf.set_text_color(224, 123, 57)
        pdf.cell(0, 6, "You:" if is_user else "Assistant:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)
        # Content — keep [N] markers so inline citations are visible in the PDF
        content = msg["content"].strip()
        pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(0, 5, _safe(content), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        # Citations — use cell() (clips, never raises) instead of multi_cell().
        # URLs are replaced with "(link)" since they are not clickable in PDFs.
        for cite in msg.get("citations") or []:
            locs = cite.get("locations") or []
            if locs:
                first = locs[0]
                loc_suffix = " (link)" if first.startswith("http") else f", {first}"
                if len(locs) > 1:
                    loc_suffix += f" +{len(locs) - 1} more"
            else:
                loc_suffix = ""
            line = _safe(f"  [{cite['num']}] {cite['source_name']}{loc_suffix}")
            pdf.set_font("Helvetica", size=8)
            pdf.set_text_color(120, 120, 120)
            pdf.multi_cell(0, 4, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_text_color(0, 0, 0)
        pdf.ln(4)

    return bytes(pdf.output())


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
        lambda m: f'<span style="color:{_ORANGE};font-weight:bold;">[{m.group(1)}]</span>',
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

    badge = f'<span style="color:{_ORANGE};font-weight:bold;">[{num}]</span>'

    if locations and locations[0].startswith("http"):
        if src_type == "YouTube" and len(locations) > 1:
            # Multiple timestamps from the same video — hyperlink each timestamp
            ts_parts = ", ".join(f"[{_ts_label(loc)}]({loc})" for loc in locations)
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
