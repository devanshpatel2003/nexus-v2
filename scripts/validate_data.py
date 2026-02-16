#!/usr/bin/env python3
"""
Validate NEXUS v2 data integrity.
Run: python scripts/validate_data.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.export_control_events import get_events_dataframe, get_event_summary
from data.universe import ALL_TICKERS, ECOSYSTEM, TICKER_NAMES
from core.rag.chunking import load_and_chunk_directory
from core.config import KNOWLEDGE_DIR


def main():
    print("=" * 60)
    print("NEXUS v2 — Data Validation")
    print("=" * 60)

    errors = []

    # 1. Event database
    print("\n[1] Event Database")
    events = get_events_dataframe()
    print(f"  Events: {len(events)}")
    print(f"  Date range: {events['date'].min()} to {events['date'].max()}")
    summary = get_event_summary()
    print(f"  By severity: {summary['by_severity']}")
    if len(events) < 10:
        errors.append(f"Expected 10+ events, got {len(events)}")

    # 2. Universe
    print("\n[2] Ticker Universe")
    print(f"  All tickers: {ALL_TICKERS}")
    print(f"  Ecosystem groups: {len(ECOSYSTEM)}")
    for group, tickers in ECOSYSTEM.items():
        print(f"    {group}: {tickers}")
    if len(ALL_TICKERS) < 10:
        errors.append(f"Expected 10+ tickers, got {len(ALL_TICKERS)}")

    # 3. Knowledge documents
    print("\n[3] Knowledge Base")
    chunks = load_and_chunk_directory(KNOWLEDGE_DIR)
    print(f"  Total chunks: {len(chunks)}")
    source_types = set(c["metadata"]["source_type"] for c in chunks)
    print(f"  Source types: {source_types}")
    if len(chunks) < 10:
        errors.append(f"Expected 10+ chunks, got {len(chunks)}")

    # 4. Document manifest
    print("\n[4] Document Manifest")
    manifest_path = os.path.join(os.path.dirname(KNOWLEDGE_DIR), "..", "knowledge", "manifests", "documents.json")
    manifest_path = os.path.normpath(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "knowledge", "manifests", "documents.json"
    ))
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"  Documents in manifest: {len(manifest)}")
    else:
        errors.append("Document manifest not found")

    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"VALIDATION FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"  ✗ {e}")
    else:
        print("VALIDATION PASSED — All checks OK")
    print("=" * 60)

    return len(errors) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
