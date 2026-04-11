from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from config.settings import GWDG_API_BASE, GWDG_API_KEY, GWDG_MODEL_NAME, MAX_HISTORY_TURNS

_REWRITE_PROMPT = """\
Given the following conversation history and a follow-up question, \
rewrite the follow-up question as a standalone question that captures \
all necessary context.

Chat history:
{history}

Follow-up question: {question}

Standalone question:"""


class ConversationMemory:
    """Stores the last N conversation turns and rewrites follow-up queries."""

    def __init__(self) -> None:
        self._turns: list[tuple[str, str]] = []  # (user_msg, assistant_msg)
        self._llm: ChatOpenAI | None = None

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def add_turn(self, user_message: str, assistant_message: str) -> None:
        self._turns.append((user_message, assistant_message))
        if len(self._turns) > MAX_HISTORY_TURNS:
            self._turns = self._turns[-MAX_HISTORY_TURNS:]

    def get_history(self) -> list[dict]:
        """Return history as a flat list of role/content dicts."""
        history = []
        for user_msg, asst_msg in self._turns:
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": asst_msg})
        return history

    def clear(self) -> None:
        self._turns.clear()

    # ------------------------------------------------------------------
    # Query rewriting
    # ------------------------------------------------------------------

    def rewrite_query(self, query: str, history: list[dict] | None = None) -> str:
        """Rewrite a follow-up question as a self-contained standalone question.

        Uses *history* when provided (from session state), otherwise falls back
        to the internally stored turns. Returns the original query unchanged if
        there is no history to resolve pronouns against.
        """
        turns = history or self.get_history()
        if not turns:
            return query

        # Build a readable chat history string from role/content dicts
        lines = []
        for msg in turns:
            role = msg.get("role", "").capitalize()
            lines.append(f"{role}: {msg.get('content', '')}")
        history_text = "\n".join(lines)

        prompt = _REWRITE_PROMPT.format(history=history_text, question=query)
        response = self._get_llm().invoke([HumanMessage(content=prompt)])
        rewritten = response.content.strip()
        return rewritten if rewritten else query

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(
                api_key=GWDG_API_KEY,
                base_url=GWDG_API_BASE,
                model=GWDG_MODEL_NAME,
                temperature=0,
            )
        return self._llm
