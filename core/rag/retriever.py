"""
RAG Retriever
Retrieves relevant context and formats it for the LLM.
"""

from typing import List, Dict, Optional
from core.rag.vector_store import search
from core.config import RETRIEVAL_TOP_K


def retrieve_context(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    source_type: Optional[str] = None,
    severity: Optional[str] = None,
) -> List[Dict]:
    """
    Retrieve relevant documents for a query.
    Optionally filter by source_type or severity.
    """
    metadata_filter = None
    filters = {}
    if source_type:
        filters["source_type"] = source_type
    if severity:
        filters["severity"] = severity

    if filters:
        if len(filters) == 1:
            key, val = list(filters.items())[0]
            metadata_filter = {key: val}
        else:
            metadata_filter = {"$and": [{k: v} for k, v in filters.items()]}

    return search(query, top_k=top_k, metadata_filter=metadata_filter)


def format_context_for_llm(hits: List[Dict]) -> str:
    """
    Format retrieved chunks into a context string for the LLM prompt.
    Includes chunk IDs for citation.
    """
    if not hits:
        return "No relevant documents found in the knowledge base."

    parts = []
    for hit in hits:
        chunk_id = hit["chunk_id"]
        text = hit["text"]
        source_type = hit["metadata"].get("source_type", "unknown")
        parts.append(f"[{chunk_id}] (source: {source_type})\n{text}")

    return "\n\n---\n\n".join(parts)
