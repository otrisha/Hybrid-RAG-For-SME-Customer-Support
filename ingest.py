"""
ingest.py
=========
Entry point for knowledge base construction.
Run once before launching the RAG system, and re-run whenever
any source document is updated.

Usage:
    python ingest.py                      # full ingest, all documents
    python ingest.py --doc BDS-FAQ-001    # single document only
    python ingest.py --dry-run            # chunk only, skip Pinecone upload

Thesis reference: Section 4.1 (Document Ingestion Pipeline).
"""

import argparse
import sys
from pathlib import Path

from config.settings import DOCUMENTS, validate_config, BM25_INDEX_PATH, CHUNKS_DIR
from ingestion.document_loader import load_document, load_all_documents
from ingestion.chunker import chunk_document, chunk_all_documents
from ingestion.indexer import BM25Index, EmbeddingModel, upsert_to_pinecone
from utils.helpers import save_json
from utils.logger import get_logger

log = get_logger("ingest")


def run_ingest(doc_id: str | None = None, dry_run: bool = False) -> None:

    log.info("=" * 60)
    log.info("  Benamdaj RAG — Knowledge Base Ingestion")
    log.info("=" * 60)

    if not dry_run and not validate_config():
        sys.exit(1)

    # ── 1. Load documents ────────────────────────────────────────
    if doc_id:
        docs = [load_document(doc_id)]
    else:
        docs = load_all_documents()

    # ── 2. Chunk documents ───────────────────────────────────────
    if doc_id:
        chunks = chunk_document(docs[0])
    else:
        chunks = chunk_all_documents(docs)

    log.info(f"Chunking complete | {len(chunks)} total chunks")

    # ── 3. Save chunks as JSON (for inspection and RAGAS context) ─
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    chunks_data = [
        {
            "chunk_id"          : c.chunk_id,
            "document_id"       : c.document_id,
            "source"            : c.source,
            "knowledge_type"    : c.knowledge_type,
            "model"             : c.model,
            "topic_category"    : c.topic_category,
            "heading"           : c.heading,
            "retrieval_priority": c.retrieval_priority,
            "token_count"       : c.token_count,
            "persona_role"      : c.persona_role,
            "text"              : c.text,
        }
        for c in chunks
    ]
    save_json(chunks_data, CHUNKS_DIR / "all_chunks.json")
    log.info(f"Chunks saved → {CHUNKS_DIR / 'all_chunks.json'}")

    # ── 4. Build and save BM25 index ─────────────────────────────
    bm25 = BM25Index(chunks)
    bm25.save(BM25_INDEX_PATH)

    if dry_run:
        log.info("Dry run complete — Pinecone upload skipped.")
        return

    # ── 5. Compute embeddings and upsert to Pinecone ─────────────
    log.info("Computing embeddings …")
    embeddings = EmbeddingModel.encode_chunks(chunks)
    log.info(f"Embeddings shape: {embeddings.shape}")

    upsert_to_pinecone(chunks, embeddings)

    log.info("=" * 60)
    log.info(f"  Ingestion complete | {len(chunks)} chunks indexed")
    log.info("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Benamdaj RAG — Knowledge Base Ingestion")
    ap.add_argument("--doc",     type=str, default=None,
                    help="Ingest a single document by ID, e.g. BDS-FAQ-001")
    ap.add_argument("--dry-run", action="store_true",
                    help="Chunk only; skip Pinecone upload (no API keys needed)")
    args = ap.parse_args()
    run_ingest(doc_id=args.doc, dry_run=args.dry_run)
