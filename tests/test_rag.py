"""
Tests for NEXUS v2 RAG components.
Tests chunking and document loading (no API key required).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from core.rag.chunking import extract_metadata, chunk_markdown, load_and_chunk_directory
from core.config import KNOWLEDGE_DIR


class TestMetadataExtraction:
    """Test metadata extraction from markdown frontmatter."""

    def test_extract_doc_id(self):
        content = "# Test Event\ndoc_id: test_123\ndate: 2024-01-01\nseverity: High"
        meta = extract_metadata(content)
        assert meta["doc_id"] == "test_123"
        assert meta["date"] == "2024-01-01"
        assert meta["severity"] == "High"
        assert meta["title"] == "Test Event"

    def test_extract_title(self):
        content = "# My Document Title\nSome content"
        meta = extract_metadata(content)
        assert meta["title"] == "My Document Title"


class TestChunking:
    """Test markdown chunking logic."""

    def test_basic_chunking(self):
        content = "# Title\ndoc_id: test\n\n## Section 1\nContent 1\n\n## Section 2\nContent 2"
        metadata = {"doc_id": "test"}
        chunks = chunk_markdown(content, metadata)
        assert len(chunks) >= 1
        assert all("chunk_id" in c for c in chunks)
        assert all("text" in c for c in chunks)

    def test_chunk_ids_unique(self):
        content = "# Title\ndoc_id: test\n\n## A\nText A\n\n## B\nText B\n\n## C\nText C"
        metadata = {"doc_id": "test"}
        chunks = chunk_markdown(content, metadata)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_metadata_inherited(self):
        content = "# Title\n\n## Section\nContent"
        metadata = {"doc_id": "test", "severity": "Critical"}
        chunks = chunk_markdown(content, metadata)
        for chunk in chunks:
            assert chunk["metadata"]["severity"] == "Critical"


class TestDocumentLoading:
    """Test loading documents from knowledge directory."""

    def test_load_knowledge_dir(self):
        if not os.path.exists(KNOWLEDGE_DIR):
            pytest.skip("Knowledge directory not found")
        chunks = load_and_chunk_directory(KNOWLEDGE_DIR)
        assert len(chunks) > 0
        assert all("source_type" in c["metadata"] for c in chunks)

    def test_source_types_assigned(self):
        if not os.path.exists(KNOWLEDGE_DIR):
            pytest.skip("Knowledge directory not found")
        chunks = load_and_chunk_directory(KNOWLEDGE_DIR)
        source_types = set(c["metadata"]["source_type"] for c in chunks)
        assert "event_db" in source_types
        assert "methodology" in source_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
