"""
Vector Store
ChromaDB-based vector storage and retrieval.
"""

import os
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from core.config import CHROMA_PERSIST_DIR, KNOWLEDGE_DIR
from core.llm.client import get_embeddings
from core.rag.chunking import load_and_chunk_directory


def get_collection(collection_name: str = "nexus_docs") -> chromadb.Collection:
    """Get or create a ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def build_index(knowledge_dir: str = None) -> int:
    """
    Build the vector index from knowledge documents.
    Returns the number of chunks indexed.
    """
    if knowledge_dir is None:
        knowledge_dir = KNOWLEDGE_DIR

    chunks = load_and_chunk_directory(knowledge_dir)

    if not chunks:
        print("No documents found to index.")
        return 0

    collection = get_collection()

    # Clear existing data
    try:
        existing = collection.count()
        if existing > 0:
            all_ids = collection.get()["ids"]
            if all_ids:
                collection.delete(ids=all_ids)
    except Exception:
        pass

    # Batch embed and add
    batch_size = 20
    total = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        embeddings = get_embeddings(texts)

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        total += len(batch)

    print(f"Indexed {total} chunks from {knowledge_dir}")
    return total


def search(
    query: str,
    top_k: int = 6,
    metadata_filter: Optional[Dict] = None,
) -> List[Dict]:
    """
    Search the vector store for relevant documents.
    Returns list of {chunk_id, text, metadata, score}.
    """
    from core.llm.client import get_embedding

    collection = get_collection()

    if collection.count() == 0:
        return []

    query_embedding = get_embedding(query)

    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }

    if metadata_filter:
        kwargs["where"] = metadata_filter

    results = collection.query(**kwargs)

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    return hits
