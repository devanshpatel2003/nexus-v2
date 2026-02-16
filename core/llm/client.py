"""
OpenAI Client Wrapper
Handles API calls for chat completion and embeddings.
"""

from openai import OpenAI
from typing import List, Dict, Optional
from core.config import get_openai_key, LLM_MODEL, EMBEDDING_MODEL


_client = None


def get_client() -> OpenAI:
    """Get or create OpenAI client (singleton)."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=get_openai_key())
    return _client


def chat_completion(
    messages: List[Dict],
    tools: Optional[List[Dict]] = None,
    model: str = LLM_MODEL,
    temperature: float = 0.3,
) -> Dict:
    """
    Call OpenAI chat completion with optional tool definitions.
    Returns the full response message dict.
    """
    client = get_client()
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message


def get_embeddings(texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
    """Get embeddings for a list of texts."""
    client = get_client()
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    """Get embedding for a single text."""
    return get_embeddings([text], model)[0]
