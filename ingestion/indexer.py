"""
ingestion/indexer.py
====================
Dual indexing: BM25 inverted index + Pinecone vector database.

  BM25Index     — in-memory sparse index backed by rank-bm25
  EmbeddingModel — singleton sentence-transformer encoder
  upsert_to_pinecone() — uploads chunk embeddings to Pinecone

Thesis reference: Section 5.2–5.3 (Retrieval Architecture).
"""

from __future__ import annotations
import pickle
import time
from pathlib import Path

import nltk
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config.settings import (
    EMBEDDING_MODEL, EMBEDDING_BATCH, EMBEDDING_DIM,
    BM25_K1, BM25_B, BM25_INDEX_PATH,
    PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_DIMENSION,
    PINECONE_METRIC, PINECONE_CLOUD, PINECONE_REGION, PINECONE_NAMESPACE,
)
from ingestion.chunker import Chunk
from utils.logger import get_logger

log = get_logger(__name__)

# Ensure NLTK data is available
for _pkg in ("punkt", "punkt_tab", "stopwords"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}" if "punkt" in _pkg else f"corpora/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

_STOPWORDS = set(nltk.corpus.stopwords.words("english"))


# ── Tokeniser ─────────────────────────────────────────────────────────────────

def tokenise_for_bm25(text: str) -> list[str]:
    """
    Tokenise text for BM25 indexing.
    No stemming — technical terms (3408, 16/14, rpm) must match exactly.
    Stopword removal only.
    """
    tokens = nltk.word_tokenize(text.lower())
    return [t for t in tokens if t.isalnum() and t not in _STOPWORDS]


# ── BM25 Index ────────────────────────────────────────────────────────────────

class BM25Index:
    """
    Wrapper around BM25Okapi that keeps the chunk list aligned with
    the token corpus so that score indices map directly to chunks.
    """

    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks
        corpus       = [tokenise_for_bm25(c.bm25_text) for c in chunks]
        self._bm25   = BM25Okapi(corpus, k1=BM25_K1, b=BM25_B)
        log.info(f"BM25Index built | {len(chunks)} documents | k1={BM25_K1} b={BM25_B}")

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        tokens = tokenise_for_bm25(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(idx, float(score)) for idx, score in ranked[:top_k] if score > 0]

    def get_chunk(self, index: int) -> Chunk:
        return self._chunks[index]

    def __len__(self) -> int:
        return len(self._chunks)

    def save(self, path: Path = BM25_INDEX_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info(f"BM25 index saved → {path}")

    @classmethod
    def load(cls, path: Path = BM25_INDEX_PATH) -> "BM25Index":
        if not path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {path}. Run: python ingest.py"
            )
        with open(path, "rb") as f:
            index = pickle.load(f)
        log.info(f"BM25 index loaded | {len(index)} chunks")
        return index


# ── Embedding Model (singleton) ───────────────────────────────────────────────

class EmbeddingModel:
    _instance: SentenceTransformer | None = None

    @classmethod
    def get(cls) -> SentenceTransformer:
        if cls._instance is None:
            log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            cls._instance = SentenceTransformer(EMBEDDING_MODEL)
            log.info(f"Embedding model ready | dim={EMBEDDING_DIM}")
        return cls._instance

    @classmethod
    def encode_chunks(cls, chunks: list[Chunk]) -> np.ndarray:
        model  = cls.get()
        texts  = [c.text for c in chunks]
        log.info(f"Encoding {len(texts)} chunks …")
        vecs   = model.encode(
            texts, batch_size=EMBEDDING_BATCH,
            normalize_embeddings=True, show_progress_bar=True
        )
        return vecs.astype(np.float32)

    @classmethod
    def encode_query(cls, query: str) -> np.ndarray:
        model = cls.get()
        vec   = model.encode([query], normalize_embeddings=True)
        return vec[0].astype(np.float32)


# ── Pinecone upsert ───────────────────────────────────────────────────────────

def _ensure_pinecone_index(pc) -> None:
    """Create the Pinecone index if it does not already exist."""
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        log.info(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' …")
        from pinecone import ServerlessSpec
        pc.create_index(
            name      = PINECONE_INDEX_NAME,
            dimension = PINECONE_DIMENSION,
            metric    = PINECONE_METRIC,
            spec      = ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        # Wait until ready
        for _ in range(30):
            status = pc.describe_index(PINECONE_INDEX_NAME).status
            if status.get("ready"):
                break
            time.sleep(2)
        log.info("Pinecone index ready.")
    else:
        log.info(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists.")


def upsert_to_pinecone(
    chunks     : list[Chunk],
    embeddings : np.ndarray,
    batch_size : int = 100,
    namespace  : str = PINECONE_NAMESPACE,
) -> None:
    """Upload chunks and their embeddings to Pinecone."""
    from pinecone import Pinecone
    pc  = Pinecone(api_key=PINECONE_API_KEY)
    _ensure_pinecone_index(pc)
    idx = pc.Index(PINECONE_INDEX_NAME)

    vectors = []
    for chunk, vec in zip(chunks, embeddings):
        vectors.append({
            "id"      : chunk.chunk_id,
            "values"  : vec.tolist(),
            "metadata": {
                "document_id"      : chunk.document_id,
                "source"           : chunk.source,
                "knowledge_type"   : chunk.knowledge_type,
                "model"            : chunk.model,
                "topic_category"   : chunk.topic_category,
                "heading"          : chunk.heading[:200],
                "retrieval_priority": chunk.retrieval_priority,
                "token_count"      : chunk.token_count,
                "persona_role"     : chunk.persona_role,
            },
        })

    total   = len(vectors)
    upserted = 0
    for i in range(0, total, batch_size):
        batch = vectors[i : i + batch_size]
        idx.upsert(vectors=batch, namespace=namespace)
        upserted += len(batch)
        log.info(f"  Pinecone upsert: {upserted}/{total}")

    log.info(f"Pinecone upsert complete: {upserted} vectors in namespace '{namespace}'")
