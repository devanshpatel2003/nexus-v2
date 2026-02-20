"""
NEXUS v2 Configuration
Centralized settings and constants.
"""

import os
import streamlit as st


def _get_key(secret_name: str) -> str | None:
    """Get an API key from Streamlit secrets or environment. Returns None if missing."""
    try:
        val = st.secrets[secret_name]
        if val:
            return val.strip()
    except Exception:
        pass
    val = os.environ.get(secret_name)
    return val.strip() if val else None


def get_openai_key() -> str:
    key = _get_key("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "OPENAI_API_KEY not found. Set it in .streamlit/secrets.toml "
            "or as an environment variable."
        )
    return key


def get_gemini_key() -> str | None:
    return _get_key("GOOGLE_API_KEY")


def get_anthropic_key() -> str | None:
    return _get_key("ANTHROPIC_API_KEY")


# Model settings
LLM_MODEL = "gpt-4o-mini"

# All available models with provider info
ALL_MODELS = {
    "GPT-4o-mini": {"id": "gpt-4o-mini", "provider": "openai"},
    "GPT-4o": {"id": "gpt-4o", "provider": "openai"},
    "GPT-4.1-mini": {"id": "gpt-4.1-mini", "provider": "openai"},
    "GPT-4.1": {"id": "gpt-4.1", "provider": "openai"},
    "Gemini 2.0 Flash": {"id": "gemini-2.0-flash", "provider": "google"},
    "Claude 3.5 Haiku": {"id": "claude-3-5-haiku-latest", "provider": "anthropic"},
}

# Backwards compat
OPENAI_MODELS = {k: v["id"] for k, v in ALL_MODELS.items() if v["provider"] == "openai"}
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# RAG settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVAL_TOP_K = 4

# Chroma settings
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "indexes", "chroma")
KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge", "processed")

# Data settings
DEFAULT_START_DATE = "2022-01-01"
DEFAULT_BENCHMARK = "SPY"
DEFAULT_EVENT_WINDOW = (-1, 5)
DEFAULT_ESTIMATION_WINDOW = 120
