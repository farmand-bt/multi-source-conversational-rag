import streamlit as st

from rag.pipeline import RAGPipeline

_MAX_HISTORY_TURNS = 10  # user+assistant pairs kept in session state


def render_chat(pipeline: RAGPipeline) -> None:
    st.header("Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Replay stored messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("citations"):
                with st.expander("Sources"):
                    for cite in msg["citations"]:
                        _render_citation(cite)

    has_sources = bool(pipeline.list_sources())
    placeholder = (
        "Ask a question about your documents…"
        if has_sources
        else "Ingest a source from the sidebar first."
    )

    if prompt := st.chat_input(placeholder, disabled=not has_sources):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Pass prior turns as history (exclude the message just appended)
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ][-_MAX_HISTORY_TURNS * 2 :]

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer = pipeline.ask(prompt, history=history)
            st.write(answer.text)
            citations = [
                {
                    "source_type": c.source_type,
                    "source_name": c.source_name,
                    "location": c.location,
                }
                for c in answer.citations
            ]
            if citations:
                with st.expander("Sources"):
                    for cite in citations:
                        _render_citation(cite)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer.text, "citations": citations}
        )

        # Trim history to avoid unbounded growth
        max_msgs = _MAX_HISTORY_TURNS * 2
        if len(st.session_state.messages) > max_msgs:
            st.session_state.messages = st.session_state.messages[-max_msgs:]


def _render_citation(cite: dict) -> None:
    src_type = cite["source_type"]
    name = cite["source_name"]
    loc = cite["location"]
    if loc:
        st.markdown(f"- **[{src_type}]** {name} — {loc}")
    else:
        st.markdown(f"- **[{src_type}]** {name}")
