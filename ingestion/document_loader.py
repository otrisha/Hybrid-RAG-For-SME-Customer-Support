"""
ingestion/document_loader.py
=============================
Extracts text from .pdf source documents preserving heading structure,
paragraph boundaries, and table content.

Heading levels are inferred from font-size ratios relative to the
dominant body-text size on each page.  Bold short lines are also
promoted to heading level 3 when no size difference is detected.

Thesis reference: Section 4.1 (Document Ingestion Pipeline).
"""

from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import statistics

import pdfplumber

from config.settings import DOCUMENTS, DATA_DIR
from utils.helpers import clean_text, table_to_text
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class DocumentBlock:
    block_type : str   # "heading" | "paragraph" | "table"
    level      : int   # heading level 1–4; 0 for non-headings
    text       : str
    style_name : str = ""


@dataclass
class LoadedDocument:
    document_id      : str
    title            : str
    source           : str
    knowledge_type   : str
    chunking_method  : str
    blocks           : list[DocumentBlock] = field(default_factory=list)
    raw_text         : str = ""


# ── Font-size helpers ─────────────────────────────────────────────────────────

def _dominant_font_size(page) -> float:
    """Return the mode font size on the page (proxy for body-text size)."""
    sizes = [round(c["size"], 1) for c in page.chars if c.get("size")]
    if not sizes:
        return 10.0
    counter = Counter(sizes)
    return counter.most_common(1)[0][0]


def _heading_level(font_size: float, body_size: float) -> int:
    """Map font-size ratio to heading level 1–3, or 0 (paragraph)."""
    if body_size <= 0:
        return 0
    ratio = font_size / body_size
    if ratio >= 1.6:
        return 1
    if ratio >= 1.3:
        return 2
    if ratio >= 1.1:
        return 3
    return 0


def _is_bold(chars: list[dict]) -> bool:
    return any("bold" in c.get("fontname", "").lower() for c in chars)


# ── Line grouping ─────────────────────────────────────────────────────────────

def _group_chars_into_lines(chars: list[dict]) -> list[list[dict]]:
    """Cluster characters into text lines by vertical position (±2pt tolerance)."""
    if not chars:
        return []
    sorted_chars = sorted(chars, key=lambda c: (round(c["top"], 0), c["x0"]))
    lines: list[list[dict]] = []
    current: list[dict] = [sorted_chars[0]]
    for ch in sorted_chars[1:]:
        if abs(ch["top"] - current[0]["top"]) <= 2.0:
            current.append(ch)
        else:
            lines.append(sorted(current, key=lambda c: c["x0"]))
            current = [ch]
    lines.append(sorted(current, key=lambda c: c["x0"]))
    return lines


def _line_text(chars: list[dict]) -> str:
    return "".join(c.get("text", "") for c in chars).strip()


def _line_avg_size(chars: list[dict]) -> float:
    sizes = [c["size"] for c in chars if c.get("size")]
    return statistics.mean(sizes) if sizes else 0.0


# ── Table bbox filter ─────────────────────────────────────────────────────────

def _in_any_bbox(char: dict, bboxes: list[tuple]) -> bool:
    """Return True if char falls inside any of the table bounding boxes."""
    x, y = char["x0"], char["top"]
    for x0, top, x1, bottom in bboxes:
        if x0 - 1 <= x <= x1 + 1 and top - 1 <= y <= bottom + 1:
            return True
    return False


# ── Core extraction ───────────────────────────────────────────────────────────

def _extract_blocks_from_pdf(pdf_path: Path) -> list[DocumentBlock]:
    """
    Open a PDF with pdfplumber and return a list of DocumentBlocks
    (tables, headings, paragraphs) in reading order.
    """
    blocks: list[DocumentBlock] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:

            # ── 1. Extract tables and record their bounding boxes ──────────
            table_bboxes: list[tuple] = []
            for table_obj in page.find_tables():
                table_bboxes.append(table_obj.bbox)
                rows = table_obj.extract()
                if not rows:
                    continue
                # Replace None cells with empty string
                clean_rows = [[cell if cell is not None else "" for cell in row]
                              for row in rows]
                text = clean_text(table_to_text(clean_rows))
                if text:
                    blocks.append(DocumentBlock(
                        block_type="table", level=0,
                        text=text, style_name="table",
                    ))

            # ── 2. Extract text characters, skipping table regions ─────────
            text_chars = [c for c in page.chars
                          if not _in_any_bbox(c, table_bboxes)]

            body_size = _dominant_font_size(page)
            lines     = _group_chars_into_lines(text_chars)

            for line_chars in lines:
                text = _line_text(line_chars)
                if not text:
                    continue

                avg_size = _line_avg_size(line_chars)
                level    = _heading_level(avg_size, body_size)

                # Promote bold short lines to heading level 3 when size is
                # the same as body text (common in PDFs with bold headings)
                if level == 0 and _is_bold(line_chars) and len(text) <= 100:
                    level = 3

                btype = "heading" if level > 0 else "paragraph"
                blocks.append(DocumentBlock(
                    block_type=btype,
                    level=level,
                    text=clean_text(text),
                    style_name=f"size-{avg_size:.1f}",
                ))

    return blocks


# ── Public API ────────────────────────────────────────────────────────────────

def load_document(document_id: str) -> LoadedDocument:
    if document_id not in DOCUMENTS:
        raise ValueError(f"Unknown document_id '{document_id}'")
    cfg      = DOCUMENTS[document_id]
    filepath = DATA_DIR / cfg["filename"]
    if not filepath.exists():
        raise FileNotFoundError(
            f"Document not found: {filepath}. Place the .pdf in {DATA_DIR}/"
        )

    log.info(f"Loading {document_id}: {filepath.name}")

    loaded = LoadedDocument(
        document_id     = document_id,
        title           = cfg["title"],
        source          = cfg["source"],
        knowledge_type  = cfg["knowledge_type"],
        chunking_method = cfg["chunking_method"],
    )

    loaded.blocks   = _extract_blocks_from_pdf(filepath)
    loaded.raw_text = "\n\n".join(b.text for b in loaded.blocks if b.text)

    log.info(
        f"  {document_id}: {len(loaded.blocks)} blocks | "
        f"{len(loaded.raw_text):,} chars"
    )
    return loaded


def load_all_documents() -> list[LoadedDocument]:
    loaded = []
    for doc_id in DOCUMENTS:
        try:
            loaded.append(load_document(doc_id))
        except FileNotFoundError as e:
            log.warning(str(e))
    if not loaded:
        raise RuntimeError(
            "No documents could be loaded. Check DATA_DIR in settings.py."
        )
    return loaded
