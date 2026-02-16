#!/usr/bin/env python3
"""
Build the NEXUS v2 vector index from knowledge documents.
Run: python scripts/build_index.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag.vector_store import build_index


def main():
    print("=" * 60)
    print("NEXUS v2 — Building Vector Index")
    print("=" * 60)

    count = build_index()

    print(f"\nDone. {count} chunks indexed.")
    print("Index saved to indexes/chroma/")


if __name__ == "__main__":
    main()
