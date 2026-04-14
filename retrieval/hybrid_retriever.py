"""
retrieval/hybrid_retriever.py
==============================
Runs BM25 sparse and Pinecone dense retrieval in parallel, then fuses
results using Reciprocal Rank Fusion (RRF).

Supports all four ablation configurations:
  Config A — bm25_only         (sparse baseline)
  Config B — dense_only        (dense baseline)
  Config C — hybrid            (proposed system, default)
  Config D — hybrid_no_faq    (hybrid without BDS-FAQ-001)

Thesis reference: Sections 5.4 & 8.4.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from config.settings import (
    TOP_K_BM25, TOP_K_DENSE, TOP_K_FINAL, RRF_K,
    RETRIEVAL_MODE, PINECONE_INDEX_NAME, PINECONE_API_KEY,
)
from ingestion.chunker import Chunk
from ingestion.indexer import BM25Index, EmbeddingModel
from retrieval.query_processor import ProcessedQuery
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class RetrievedChunk:
    chunk            : Chunk
    rrf_score        : float
    bm25_rank        : int | None
    dense_rank       : int | None
    retrieval_source : str   # "bm25_only" | "dense_only" | "both"


def _rrf(
    bm25_results : list[tuple[int, float]],
    dense_results: list[tuple[str, float]],
    bm25_chunks  : list[Chunk],
    k            : int = RRF_K,
) -> list[tuple[str, float, int | None, int | None]]:
    rrf_scores : dict[str, float] = {}
    bm25_ranks : dict[str, int]   = {}
    dense_ranks: dict[str, int]   = {}

    for rank, (chunk, _) in enumerate(zip(bm25_chunks, bm25_results), start=1):
        cid = chunk.chunk_id
        rrf_scores[cid]  = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
        bm25_ranks[cid]  = rank

    for rank, (cid, _) in enumerate(dense_results, start=1):
        rrf_scores[cid]  = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
        dense_ranks[cid] = rank

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(cid, score, bm25_ranks.get(cid), dense_ranks.get(cid)) for cid, score in ranked]


class HybridRetriever:
    def __init__(self, bm25_index: BM25Index, mode: str = RETRIEVAL_MODE):
        self.bm25     = bm25_index
        self.mode     = mode
        self.embedder = EmbeddingModel.get()
        self._pc_index = None
        log.info(f"HybridRetriever ready | mode={mode} | chunks={len(bm25_index)}")

    def _pinecone(self):
        if self._pc_index is None:
            from pinecone import Pinecone
            self._pc_index = Pinecone(api_key=PINECONE_API_KEY).Index(PINECONE_INDEX_NAME)
        return self._pc_index

    def _dense_search(self, vec: np.ndarray, top_k: int,
                      pfilter: dict | None, exclude_faq: bool,
                      namespace: str = "benamdaj-kb") -> list[tuple[str, float]]:
        eff_filter = dict(pfilter) if pfilter else {}
        if exclude_faq:
            faq_x = {"source": {"$ne": "staff_faq"}}
            eff_filter = ({"$and": [eff_filter, faq_x]} if eff_filter else faq_x)
        kw = dict(vector=vec.tolist(), top_k=top_k, include_metadata=True, namespace=namespace)
        if eff_filter:
            kw["filter"] = eff_filter
        return [(m.id, m.score) for m in self._pinecone().query(**kw).matches]

    def _bm25_search(self, query: str, top_k: int,
                     model_filter: str | None, exclude_faq: bool) -> list[tuple[int, float]]:
        raw = self.bm25.search(query, top_k=top_k * 3)
        filtered = []
        for idx, score in raw:
            c = self.bm25.get_chunk(idx)
            if model_filter and c.model not in (model_filter, "All"):
                continue
            if exclude_faq and c.source == "staff_faq":
                continue
            filtered.append((idx, score))
            if len(filtered) >= top_k:
                break
        return filtered

    def retrieve(self, pq: ProcessedQuery, top_k: int = TOP_K_FINAL,
                 namespace: str = "benamdaj-kb") -> list[RetrievedChunk]:
        exclude_faq = (self.mode == "hybrid_no_faq")
        bm25_res: list[tuple[int, float]] = []
        dense_res: list[tuple[str, float]] = []

        if self.mode in ("bm25_only", "hybrid", "hybrid_no_faq"):
            bm25_res = self._bm25_search(pq.cleaned_query, TOP_K_BM25,
                                          pq.detected_model, exclude_faq)
        if self.mode in ("dense_only", "hybrid", "hybrid_no_faq"):
            vec = EmbeddingModel.encode_query(pq.cleaned_query)
            dense_res = self._dense_search(vec, TOP_K_DENSE,
                                           pq.pinecone_filter, exclude_faq, namespace)

        bm25_chunk_list = [self.bm25.get_chunk(i) for i, _ in bm25_res]

        if self.mode == "bm25_only":
            fused = [(bm25_chunk_list[r].chunk_id, 1.0/(RRF_K+r+1), r+1, None)
                     for r in range(min(top_k, len(bm25_chunk_list)))]
        elif self.mode == "dense_only":
            fused = [(cid, sc, None, r+1) for r, (cid, sc) in enumerate(dense_res[:top_k])]
        else:
            fused = _rrf(bm25_res, dense_res, bm25_chunk_list, k=RRF_K)[:top_k]

        # Build chunk lookup
        lookup: dict[str, Chunk] = {
            self.bm25.get_chunk(i).chunk_id: self.bm25.get_chunk(i)
            for i in range(len(self.bm25))
        }

        results: list[RetrievedChunk] = []
        for cid, score, br, dr in fused:
            chunk = lookup.get(cid)
            if chunk is None:
                log.warning(f"Chunk {cid} not found in BM25 store — skipping")
                continue
            source = ("both" if br is not None and dr is not None else
                      "bm25_only" if br is not None else "dense_only")
            results.append(RetrievedChunk(chunk=chunk, rrf_score=score,
                                           bm25_rank=br, dense_rank=dr,
                                           retrieval_source=source))
        log.debug(f"Retrieved {len(results)} chunks | mode={self.mode}")
        return results
