"""
NEXUS v2 Configuration
Centralized settings and constants.
"""

import os
import streamlit as st


def get_openai_key() -> str:
    """Get OpenAI API key from Streamlit secrets or environment."""
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError(
                "OPENAI_API_KEY not found. Set it in .streamlit/secrets.toml "
                "or as an environment variable."
            )
        return key


# Model settings
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# RAG settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVAL_TOP_K = 6

# Chroma settings
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "indexes", "chroma")
KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge", "processed")

# Data settings
DEFAULT_START_DATE = "2022-01-01"
DEFAULT_BENCHMARK = "SPY"
DEFAULT_EVENT_WINDOW = (-1, 5)
DEFAULT_ESTIMATION_WINDOW = 120
