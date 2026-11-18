from typing import Any, Dict
from app.tools.base import ToolExecutionError
from app.vector_store import FAISSVectorMemory


class FAISSRagTool:
    """
    Semantic Vector Memory Retrieval Tool using SentenceTransformers + FAISS.
    Queries the embedded knowledge base accumulated during the run.
    """

    name = "faiss_rag_memory"
    description = (
        "Queries the FAISS vector store using SentenceTransformer embeddings (all-MiniLM-L6-v2) "
        "to retrieve the most semantically relevant evidence chunks gathered from web search and crawling."
    )

    def __init__(self, memory: FAISSVectorMemory):
        self.memory = memory

    async def execute(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            raise ToolExecutionError(
                self.name, "InvalidInputError", "FAISS semantic query cannot be empty."
            )

        if self.memory.total_chunks == 0:
            raise ToolExecutionError(
                self.name,
                "EmptyVectorIndexError",
                "FAISS index is currently empty. Run serper_web_search or tavily_web_crawler first to populate memory.",
            )

        matches = self.memory.search(cleaned_query, top_k=top_k)
        return {
            "embedding_model": self.memory.model_name,
            "index_type": "FAISS IndexFlatIP (Cosine Similarity)",
            "total_indexed_chunks": self.memory.total_chunks,
            "query": cleaned_query,
            "top_matches": matches,
        }
