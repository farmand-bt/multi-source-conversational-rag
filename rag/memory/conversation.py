# Milestone 5: Implement chat history management and history-aware query rewriting.
# Keep the last MAX_HISTORY_TURNS turns from config.settings.
# Query rewriting uses one LLM call to resolve pronoun references from context.


class ConversationMemory:
    def add_turn(self, user_message: str, assistant_message: str) -> None:
        raise NotImplementedError("ConversationMemory implemented in Milestone 5")

    def rewrite_query(self, query: str) -> str:
        raise NotImplementedError("ConversationMemory implemented in Milestone 5")

    def get_history(self) -> list[dict]:
        raise NotImplementedError("ConversationMemory implemented in Milestone 5")
