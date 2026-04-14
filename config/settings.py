"""
config/settings.py
==================
Benamdaj Dredging Solutions Ltd — Hybrid RAG System
BDS-RAG-001 | Centralised configuration

All environment variables, model names, index settings, and
system-wide constants are defined here. Never hardcode API keys
in source files — use the .env file.

Thesis reference: Chapter 7.2 (Technology Stack).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Base paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / "config" / ".env")

# ── Project identity ──────────────────────────────────────────────────────────
PROJECT_NAME = "Benamdaj Dredging Solutions Ltd — Hybrid RAG Customer Support"
DOCUMENT_REF = "BDS-RAG-001"
COMPANY_NAME = "Benamdaj Dredging Solutions Ltd."
COMPANY_RC   = "RC: 1653951"
AUTHOR       = "Patricia Orji — MSc AI & Data Science, University of Wolverhampton"

# ── API keys (from .env) ──────────────────────────────────────────────────────
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY",   "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")

# ── Pinecone ──────────────────────────────────────────────────────────────────
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "benamdaj-rag")
PINECONE_CLOUD      = os.getenv("PINECONE_CLOUD",  "aws")
PINECONE_REGION     = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_DIMENSION  = 384       # all-MiniLM-L6-v2 output dimension
PINECONE_METRIC     = "cosine"
PINECONE_NAMESPACE  = "benamdaj-kb"

# ── Embedding model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384
EMBEDDING_BATCH = 32

# ── OpenAI generator ──────────────────────────────────────────────────────────
OPENAI_MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_TOKENS    = 768
TEMPERATURE   = 0.1     # low temp → factual, grounded responses

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_BM25   = 7        # BM25 candidates before fusion
TOP_K_DENSE  = 7        # Pinecone candidates before fusion
TOP_K_FINAL  = 5        # chunks passed to generator after RRF
RRF_K        = 60       # Reciprocal Rank Fusion smoothing constant (Robertson 2009)

# BM25 hyperparameters (Robertson & Zaragoza, 2009 defaults)
BM25_K1 = 1.5
BM25_B  = 0.75

# Ablation/retrieval mode: "hybrid" | "bm25_only" | "dense_only" | "hybrid_no_faq"
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")

# ── Chunking ──────────────────────────────────────────────────────────────────
MAX_CHUNK_TOKENS = 600   # split if larger than this
MIN_CHUNK_TOKENS = 80    # merge if smaller than this

# ── Knowledge base documents ──────────────────────────────────────────────────
# filename: the .pdf file should be placed in DATA_DIR
DATA_DIR = BASE_DIR / "data" / "documents"

DOCUMENTS = {
    "BDS-SPEC-001": {
        "filename"       : "Benamdaj_Product_Specification_Manual.pdf",
        "title"          : "Product Specification Manual",
        "source"         : "product_spec",
        "knowledge_type" : "explicit_structured",
        "chunking_method": "section",
        "description"    : "Specifications for all four dredger models",
    },
    "BDS-OM-001": {
        "filename"       : "Benamdaj_OM_Manual.pdf",
        "title"          : "Operations and Maintenance Manual",
        "source"         : "om_manual",
        "knowledge_type" : "explicit_procedural",
        "chunking_method": "procedure",
        "description"    : "Startup, shutdown, maintenance schedules, safety procedures",
    },
    "BDS-TSG-001": {
        "filename"       : "Benamdaj_Troubleshooting_Guide.pdf",
        "title"          : "Troubleshooting Guide",
        "source"         : "troubleshooting",
        "knowledge_type" : "explicit_diagnostic",
        "chunking_method": "fault_block",
        "description"    : "25 fault blocks covering all major system failures",
    },
    "BDS-FAQ-001": {
        "filename"       : "BDS-FAQ-001_Staff_Interview_FAQs.pdf",
        "title"          : "Staff Interview FAQs",
        "source"         : "staff_faq",
        "knowledge_type" : "tacit_elicited",
        "chunking_method": "qa_pair",
        "description"    : "56 Q&A pairs from structured staff interviews (6 staff personas, 8 topic categories)",
    },
    "BDS-PL-001": {
        "filename"       : "Benamdaj_price_list.pdf",
        "title"          : "Price List",
        "source"         : "price_list",
        "knowledge_type" : "explicit_structured",
        "chunking_method": "section",
        "description"    : "Equipment purchase prices, spare parts pricing, service packages, and payment terms",
    },
}

# ── Model vocabulary for query routing ───────────────────────────────────────
MODEL_VOCABULARY = {
    "Model 1": ["16/14", "16 inch", "16-inch", "cutter suction", "csd",
                 "model 1", "weichai 6170", "x6170", "largest dredger", "bds-csd"],
    "Model 2": ["14/12", "14 inch", "14-inch", "amphibious", "multifunctional",
                 "model 2", "amphibious dredger", "bds-amd"],
    "Model 3": ["12/10", "12 inch", "12-inch", "bucket chain", "bcd",
                 "model 3", "cat 3408", "caterpillar 3408", "3408", "bds-bcd"],
    "Model 4": ["10/10", "10 inch", "10-inch", "jet suction", "jsd",
                 "model 4", "weichai wp13", "wp13", "pto", "bds-jsd"],
}

# ── Paths ─────────────────────────────────────────────────────────────────────
CHUNKS_DIR      = BASE_DIR / "data" / "chunks"
BM25_INDEX_PATH = BASE_DIR / "data" / "bm25_index.pkl"
LOG_DIR         = BASE_DIR / "logs"

for _d in [DATA_DIR, CHUNKS_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Evaluation targets ────────────────────────────────────────────────────────
# Thesis reference: Tables 6 & 7
EVAL_TARGETS = {
    "recall_at_5"         : 0.80,
    "precision_at_5"      : 0.40,   # adjusted for 5-doc corpus (cross-doc retrieval is expected)
    "mrr"                 : 0.70,
    "ragas_faithfulness"  : 0.80,
    "ragas_answer_rel"    : 0.75,
    "ragas_ctx_precision" : 0.65,
    "ragas_ctx_recall"    : 0.75,
    "max_latency_seconds" : 8.0,
}

# ── Config validation ─────────────────────────────────────────────────────────
def validate_config() -> bool:
    errors = []
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY not set in .env")
    if not PINECONE_API_KEY:
        errors.append("PINECONE_API_KEY not set in .env")
    if errors:
        for e in errors:
            print(f"[CONFIG ERROR] {e}")
        return False
    return True
