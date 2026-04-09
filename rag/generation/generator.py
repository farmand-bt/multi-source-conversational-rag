from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config.settings import GWDG_API_BASE, GWDG_API_KEY, GWDG_MODEL_NAME
from rag.ingestion.base import Document

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly from the provided context.\n"
    "After every factual claim, cite its source using exactly one of these formats:\n"
    "  • PDF sources:     [PDF: filename.pdf, page N]\n"
    "  • Web pages:       [Web: page title, URL]\n"
    "  • YouTube videos:  [YouTube: video title, MM:SS]\n"
    "Match the citation type to the source type shown in the context header.\n"
    "If the context does not contain enough information to answer, say so clearly."
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
            header = self._format_header(doc)
            parts.append(f"{header}\n{doc.text}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _format_header(doc: Document) -> str:
        """Format the source header shown to the LLM so it knows which citation tag to use."""
        if doc.source_type == "pdf":
            page = doc.page_number or "?"
            return f"[PDF: {doc.source_name}, page {page}]"
        if doc.source_type == "youtube":
            ts = doc.timestamp or "0:00"
            return f"[YouTube: {doc.source_name}, {ts}]"
        # web
        url = doc.url or doc.source_name
        return f"[Web: {doc.source_name}, {url}]"
