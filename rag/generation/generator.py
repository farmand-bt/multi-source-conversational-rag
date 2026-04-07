from rag.ingestion.base import Document

# Milestone 3: Implement LLM answer generation via GWDG API (OpenAI-compatible).
# Prompt must instruct LLM to cite sources as [Source: name, location].
# The Streamlit UI parses these citation strings for clickable highlights.


class Generator:
    def generate(self, query: str, context_docs: list[Document]) -> str:
        raise NotImplementedError("Generator implemented in Milestone 3")
