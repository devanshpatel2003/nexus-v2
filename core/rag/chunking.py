"""
Document Chunking
Markdown-aware chunking with metadata extraction.
"""

import os
import re
from typing import List, Dict
from core.config import CHUNK_SIZE, CHUNK_OVERLAP


def extract_metadata(content: str) -> Dict[str, str]:
    """Extract YAML-like metadata from markdown frontmatter."""
    metadata = {}
    lines = content.split("\n")
    for line in lines:
        if line.startswith("# "):
            metadata["title"] = line[2:].strip()
        for field in ["doc_id", "date", "severity", "event_type", "tickers", "source"]:
            if line.startswith(f"{field}:"):
                metadata[field] = line.split(":", 1)[1].strip()
    return metadata


def chunk_markdown(content: str, metadata: Dict[str, str]) -> List[Dict]:
    """
    Split markdown content into chunks by heading sections.
    Each chunk inherits document metadata for filtering.
    """
    # Split by ## headings
    sections = re.split(r'\n(?=## )', content)

    chunks = []
    current_text = ""

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # If adding this section stays under limit, merge
        if len(current_text) + len(section) < CHUNK_SIZE:
            current_text += "\n\n" + section if current_text else section
        else:
            # Save current chunk
            if current_text:
                chunks.append({
                    "text": current_text.strip(),
                    "metadata": {**metadata},
                })
            current_text = section

    # Don't forget the last chunk
    if current_text.strip():
        chunks.append({
            "text": current_text.strip(),
            "metadata": {**metadata},
        })

    # Assign chunk IDs
    for i, chunk in enumerate(chunks):
        doc_id = metadata.get("doc_id", "unknown")
        chunk["chunk_id"] = f"{doc_id}:chunk_{i}"
        chunk["metadata"]["chunk_id"] = chunk["chunk_id"]

    return chunks


def load_and_chunk_directory(directory: str) -> List[Dict]:
    """Load all markdown files from a directory tree and chunk them."""
    all_chunks = []

    for root, dirs, files in os.walk(directory):
        for fname in sorted(files):
            if not fname.endswith(".md"):
                continue

            filepath = os.path.join(root, fname)
            with open(filepath, "r") as f:
                content = f.read()

            metadata = extract_metadata(content)
            metadata["source_file"] = fname

            # Determine source type from path
            if "events" in root:
                metadata["source_type"] = "event_db"
            elif "methodology" in root:
                metadata["source_type"] = "methodology"
            elif "case_study" in root:
                metadata["source_type"] = "case_study"
            else:
                metadata["source_type"] = "other"

            chunks = chunk_markdown(content, metadata)
            all_chunks.extend(chunks)

    return all_chunks
