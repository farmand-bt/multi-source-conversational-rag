from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config.settings import GWDG_API_BASE, GWDG_API_KEY, GWDG_MODEL_NAME
from rag.ingestion.base import Document

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly from the provided context.\n"
    "After every factual claim, cite its source using exactly this format: "
    "[Source: <source_name>, <location>]\n"
    "  • For PDFs, location = 'page N'\n"
    "  • For videos, location = 'timestamp HH:MM:SS'\n"
    "  • For web pages, location = the URL\n"
    "If the context does not contain enough information, say so clearly."
)

_USER_TEMPLATE = "Context:\n{context}\n\nQuestion: {query}"


class Generator:
    """Generates answers via an OpenAI-compatible LLM (GWDG endpoint)."""

    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            api_key=GWDG_API_KEY,
            base_url=GWDG_API_BASE,
            model=GWDG_MODEL_NAME,
            temperature=0,
        )

    def generate(
        self,
        query: str,
        context_docs: list[Document],
        history: list[dict] | None = None,
    ) -> str:
        context = self._build_context(context_docs)
        messages: list = [SystemMessage(content=_SYSTEM_PROMPT)]

        for turn in history or []:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=_USER_TEMPLATE.format(context=context, query=query)))
        response = self._llm.invoke(messages)
        return response.content

    def _build_context(self, docs: list[Document]) -> str:
        parts = []
        for doc in docs:
            location = self._format_location(doc)
            parts.append(f"[Source: {doc.source_name}, {location}]\n{doc.text}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _format_location(doc: Document) -> str:
        if doc.page_number is not None:
            return f"page {doc.page_number}"
        if doc.timestamp is not None:
            return doc.timestamp
        if doc.url is not None:
            return doc.url
        return "unknown"
