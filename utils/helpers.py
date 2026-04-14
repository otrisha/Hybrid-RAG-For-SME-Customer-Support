"""
utils/helpers.py
================
Shared utility functions for the Benamdaj Hybrid RAG pipeline.
"""

import re
import json
import hashlib
from pathlib import Path
from typing import Any


def slugify(text: str) -> str:
    """Convert text to lowercase slug for use as chunk IDs."""
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[\s_-]+", "-", text).strip("-")


def generate_chunk_id(document_id: str, index: int, label: str = "") -> str:
    """Create a stable, human-readable chunk ID."""
    base = f"{document_id}-{index:03d}"
    if label:
        base += f"-{slugify(label)[:40]}"
    return base


def count_tokens(text: str) -> int:
    """Approximate token count via whitespace splitting."""
    return len(text.split())


def clean_text(text: str) -> str:
    """
    Normalise extracted document text:
    - Collapse multiple blank lines
    - Remove trailing whitespace
    - Normalise Unicode punctuation to ASCII
    """
    text = text.replace("\u2014", "--").replace("\u2013", "-")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u00b0", " degrees")
    text = text.replace("\u00b3", "3").replace("\u00b2", "2")
    text = text.replace("\x0c", "\n")
    lines = [line.rstrip() for line in text.splitlines()]
    collapsed = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return collapsed.strip()


def table_to_text(rows: list[list[str]]) -> str:
    """Convert a table to pipe-delimited plain text."""
    if not rows:
        return ""
    return "\n".join(" | ".join(cell.strip() for cell in row) for row in rows)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fingerprint(text: str) -> str:
    """SHA-256 fingerprint of text — used for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
