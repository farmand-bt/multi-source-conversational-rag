# Central orchestrator for the RAG pipeline.
#
# Write path (Milestone 2):
#   ingest(source) -> Ingestor -> Chunker -> Embedder -> ChromaStore
#
# Read path (Milestone 3+):
#   query(text) -> ConversationMemory.rewrite_query -> Retriever -> Generator
#
# Milestone 5 adds ConversationMemory.add_turn after each query.


class RAGPipeline:
    def ingest(self, source: str, source_type: str) -> None:
        raise NotImplementedError("Ingestion pipeline implemented in Milestone 2")

    def query(self, text: str) -> str:
        raise NotImplementedError("Query pipeline implemented in Milestone 3")
