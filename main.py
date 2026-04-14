"""
main.py
=======
Flask web application and CLI entry point for the Benamdaj Hybrid RAG
customer support system.

Flask dev server (port 5000):
    python main.py
    python main.py --mode bm25_only

CLI interactive mode:
    python main.py --cli
    python main.py --cli --mode dense_only

Ingestion runs automatically on first launch if the knowledge base
has not been built yet.

Thesis reference: Section 7 (Deployment).
"""

from __future__ import annotations
import argparse
import sys

from flask import Flask, jsonify, render_template, request

from config.settings import BM25_INDEX_PATH, PINECONE_NAMESPACE, RETRIEVAL_MODE, validate_config
from generation.generator import generate
from ingestion.indexer import BM25Index, EmbeddingModel
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.query_processor import process_query
from utils.logger import get_logger

log = get_logger("main")

app = Flask(__name__)

_retriever: HybridRetriever | None = None


def _pinecone_index_exists() -> bool:
    """Return True if the Pinecone index is present and reachable."""
    try:
        from pinecone import Pinecone
        from config.settings import PINECONE_API_KEY, PINECONE_INDEX_NAME
        pc = Pinecone(api_key=PINECONE_API_KEY)
        return PINECONE_INDEX_NAME in [idx.name for idx in pc.list_indexes()]
    except Exception as exc:
        log.warning(f"Pinecone check failed: {exc}")
        return False


def _ensure_knowledge_base() -> None:
    """Run ingestion if the BM25 index or Pinecone index is missing."""
    bm25_ready     = BM25_INDEX_PATH.exists()
    pinecone_ready = _pinecone_index_exists()

    if bm25_ready and pinecone_ready:
        return

    if bm25_ready and not pinecone_ready:
        log.warning("BM25 index exists but Pinecone index is missing — deleting stale BM25 and re-ingesting …")
        BM25_INDEX_PATH.unlink(missing_ok=True)
    else:
        log.info("Knowledge base not found — running ingestion pipeline …")

    from ingest import run_ingest
    run_ingest()
    log.info("Ingestion complete.")


def get_retriever(mode: str = RETRIEVAL_MODE) -> HybridRetriever:
    """Build the knowledge base if needed, then return a cached HybridRetriever."""
    global _retriever
    _ensure_knowledge_base()
    if _retriever is None or _retriever.mode != mode:
        log.info(f"Loading retriever | mode={mode}")
        bm25 = BM25Index.load()
        EmbeddingModel.get()
        _retriever = HybridRetriever(bm25, mode=mode)
    return _retriever


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    mode = request.args.get("mode", RETRIEVAL_MODE)
    return render_template("index.html", mode=mode)


@app.route("/chat", methods=["POST"])
def chat():
    data  = request.get_json(force=True)
    query = (data.get("query") or data.get("message") or "").strip()
    mode  = data.get("mode", RETRIEVAL_MODE)

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        retriever = get_retriever(mode)
        pq        = process_query(query)
        retrieved = retriever.retrieve(pq, namespace=PINECONE_NAMESPACE)
        response  = generate(pq, retrieved, mode=mode)
    except Exception as exc:
        import traceback
        log.error(f"Chat error: {exc}\n{traceback.format_exc()}")
        return jsonify({"error": "Internal server error", "detail": str(exc)}), 500

    return jsonify({
        "answer"          : response.answer,
        "sources"         : response.sources_cited,
        "is_fallback"     : response.is_fallback,
        "detected_model"  : response.detected_model,
        "topic_category"  : response.topic_category,
        "latency_seconds" : round(response.latency_seconds, 2),
        "retrieval_mode"  : response.retrieval_mode,
    })


@app.route("/health")
def health():
    try:
        r = get_retriever()
        return jsonify({"status": "ok", "chunks": len(r.bm25), "mode": r.mode})
    except Exception as exc:
        return jsonify({"status": "error", "detail": str(exc)}), 500


# ── CLI interactive mode ──────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║   Benamdaj Dredging Solutions Ltd.                          ║
║   AI-Powered Technical Customer Support                     ║
║   Hybrid RAG System — BDS-RAG-001                          ║
║   MSc AI & Data Science | University of Wolverhampton       ║
╚══════════════════════════════════════════════════════════════╝
Type your question and press Enter. Type 'quit' to exit.
"""


def run_interactive(mode: str = RETRIEVAL_MODE) -> None:
    if not validate_config():
        sys.exit(1)
    retriever = get_retriever(mode)
    print(BANNER)
    print(f"  Retrieval mode: {mode.upper()}")
    print(f"  Knowledge base: {len(retriever.bm25)} chunks\n")

    while True:
        try:
            query = input("Customer: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Session ended.")
            break

        pq        = process_query(query)
        retrieved = retriever.retrieve(pq, namespace=PINECONE_NAMESPACE)
        response  = generate(pq, retrieved, mode=mode)

        print(f"\nAssistant: {response.answer}")
        if response.sources_cited:
            print(f"[Sources: {', '.join(response.sources_cited)}]")
        if response.is_fallback:
            print("[Note: This query could not be answered from the current documentation.]")
        print(
            f"[Latency: {response.latency_seconds:.2f}s"
            f" | Model: {pq.detected_model or 'All'}"
            f" | Topic: {pq.topic_category}]\n"
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Benamdaj Hybrid RAG Customer Support")
    ap.add_argument(
        "--mode", default=RETRIEVAL_MODE,
        choices=["hybrid", "bm25_only", "dense_only", "hybrid_no_faq"],
        help="Retrieval mode (ablation configuration)",
    )
    ap.add_argument(
        "--cli", action="store_true",
        help="Run interactive CLI session instead of the Flask web server",
    )
    ap.add_argument("--port", type=int, default=5000, help="Flask server port")
    ap.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = ap.parse_args()

    if args.cli:
        run_interactive(mode=args.mode)
    else:
        if not validate_config():
            sys.exit(1)
        # Pre-load the retriever before accepting requests
        get_retriever(args.mode)
        app.run(host="0.0.0.0", port=args.port, debug=args.debug)
