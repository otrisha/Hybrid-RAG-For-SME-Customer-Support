"""
evaluation/ragas_evaluator.py
==============================
Runs Recall@5, Precision@5, MRR, and RAGAS metrics (faithfulness,
answer relevancy, context precision, context recall).

Supports all four ablation configurations (A–D).

Thesis reference: Sections 8.2–8.4.
"""

from __future__ import annotations
import csv
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config.settings import EVAL_TARGETS, LOG_DIR, OPENAI_API_KEY
from evaluation.eval_queries import EVAL_QUERIES, EvalQuery, balanced_sample
from generation.generator import generate, RAGResponse
from ingestion.indexer import BM25Index, EmbeddingModel
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.query_processor import process_query
from utils.logger import get_logger

log = get_logger(__name__)
RESULTS_DIR = LOG_DIR / "evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class QueryResult:
    query_id           : str
    query              : str
    expected_model     : str
    topic_category     : str
    relevant_doc       : str
    answer             : str
    sources_cited      : str
    is_fallback        : bool
    has_citations      : bool
    retrieved_doc_ids  : str
    hit_at_1           : bool
    hit_at_5           : bool
    rr_score           : float
    latency_seconds    : float
    retrieval_mode     : str
    ragas_faithfulness : float | None = None
    ragas_answer_rel   : float | None = None
    ragas_ctx_precision: float | None = None
    ragas_ctx_recall   : float | None = None


def _compute_retrieval_metrics(qr: QueryResult, relevant: str,
                                retrieved_doc_ids: list[str]) -> None:
    hits = [1 if relevant in did else 0 for did in retrieved_doc_ids]
    qr.hit_at_1 = bool(hits and hits[0])
    qr.hit_at_5 = any(hits[:5])
    qr.rr_score = 0.0
    for rank, hit in enumerate(hits[:5], start=1):
        if hit:
            qr.rr_score = 1.0 / rank
            return


def _run_ragas(questions, answers, contexts, ground_truths) -> list[dict]:
    try:
        from ragas import evaluate
        from ragas.metrics import (faithfulness, answer_relevancy,
                                    context_precision, context_recall)
        from datasets import Dataset
        from langchain_openai import ChatOpenAI
        dataset = Dataset.from_dict({"question": questions, "answer": answers,
                                     "contexts": contexts, "ground_truth": ground_truths})
        result = evaluate(dataset,
                          metrics=[faithfulness, answer_relevancy,
                                   context_precision, context_recall],
                          llm=ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY))
        return result.to_pandas().to_dict("records")
    except Exception as exc:
        log.warning(f"RAGAS skipped: {exc}")
        return [{}] * len(questions)


def run_evaluation(mode: str = "hybrid", max_queries: int | None = None,
                   per_doc: int | None = None,
                   use_ragas: bool = True) -> pd.DataFrame:
    if per_doc is not None:
        queries = balanced_sample(per_doc=per_doc)
    elif max_queries is not None:
        queries = EVAL_QUERIES[:max_queries]
    else:
        queries = EVAL_QUERIES
    log.info(f"Evaluation | mode={mode} | queries={len(queries)}")
    bm25      = BM25Index.load()
    EmbeddingModel.get()
    retriever = HybridRetriever(bm25, mode=mode)
    results   : list[QueryResult] = []

    for eq in tqdm(queries, desc=f"[{mode}]"):
        pq        = process_query(eq.query)
        retrieved = retriever.retrieve(pq)
        response  = generate(pq, retrieved, mode=mode)
        doc_ids   = [rc.chunk.document_id for rc in retrieved]
        qr = QueryResult(
            query_id=eq.query_id, query=eq.query, expected_model=eq.expected_model,
            topic_category=eq.topic_category, relevant_doc=eq.relevant_doc,
            answer=response.answer, sources_cited=", ".join(response.sources_cited),
            is_fallback=response.is_fallback, has_citations=response.has_citations,
            retrieved_doc_ids=", ".join(doc_ids),
            hit_at_1=False, hit_at_5=False, rr_score=0.0,
            latency_seconds=response.latency_seconds, retrieval_mode=mode,
        )
        _compute_retrieval_metrics(qr, eq.relevant_doc, doc_ids)
        results.append(qr)

    if use_ragas:
        log.info("Running RAGAS …")
        questions     = [r.query for r in results]
        answers       = [r.answer for r in results]
        contexts      = [[rc.chunk.text for rc in retriever.retrieve(process_query(q))]
                         for q in tqdm(questions, desc="Re-retrieving")]
        ground_truths = [eq.ground_truth for eq in queries]
        for qr, ragas in zip(results, _run_ragas(questions, answers, contexts, ground_truths)):
            qr.ragas_faithfulness  = ragas.get("faithfulness")
            qr.ragas_answer_rel    = ragas.get("answer_relevancy")
            qr.ragas_ctx_precision = ragas.get("context_precision")
            qr.ragas_ctx_recall    = ragas.get("context_recall")

    df = pd.DataFrame([asdict(r) for r in results])
    ts = time.strftime("%Y%m%d_%H%M%S")
    df.to_csv(RESULTS_DIR / f"eval_{mode}_{ts}.csv", index=False)
    _print_summary(df, mode)
    return df


def _print_summary(df: pd.DataFrame, mode: str) -> None:
    T   = EVAL_TARGETS
    n   = len(df)
    r5  = df["hit_at_5"].mean()
    p5  = df.apply(lambda r: sum(1 for d in r["retrieved_doc_ids"].split(", ")
                                  if d == r["relevant_doc"]) / 5, axis=1).mean()
    mrr = df["rr_score"].mean()
    fb  = df["is_fallback"].mean()
    ci  = df["has_citations"].mean()
    lat = df["latency_seconds"].mean()
    faith = df["ragas_faithfulness"].mean() if "ragas_faithfulness" in df and df["ragas_faithfulness"].notna().any() else None
    ar    = df["ragas_answer_rel"].mean()    if "ragas_answer_rel" in df and df["ragas_answer_rel"].notna().any() else None
    cp    = df["ragas_ctx_precision"].mean() if "ragas_ctx_precision" in df and df["ragas_ctx_precision"].notna().any() else None
    cr    = df["ragas_ctx_recall"].mean()    if "ragas_ctx_recall" in df and df["ragas_ctx_recall"].notna().any() else None

    pf = lambda v, t: ("PASS ✓" if v is not None and v >= t else ("N/A  " if v is None else "FAIL ✗"))
    sep = "=" * 58
    print(f"\n{sep}")
    print(f"  BENAMDAJ RAG EVALUATION — {mode.upper()} | n={n}")
    print(sep)
    print(f"  {'Recall@5':<30} {r5:.3f}  target≥{T['recall_at_5']}  {pf(r5,T['recall_at_5'])}")
    print(f"  {'Precision@5':<30} {p5:.3f}  target≥{T['precision_at_5']}  {pf(p5,T['precision_at_5'])}")
    print(f"  {'MRR':<30} {mrr:.3f}  target≥{T['mrr']}  {pf(mrr,T['mrr'])}")
    print(f"  {'Fallback rate':<30} {fb:.3f}")
    print(f"  {'Citation rate':<30} {ci:.3f}")
    print(f"  {'Mean latency (s)':<30} {lat:.2f}  target≤8.0")
    print(sep)
    for name, val, key in [
        ("Faithfulness",      faith, "ragas_faithfulness"),
        ("Answer Relevancy",  ar,    "ragas_answer_rel"),
        ("Context Precision", cp,    "ragas_ctx_precision"),
        ("Context Recall",    cr,    "ragas_ctx_recall"),
    ]:
        vs = f"{val:.3f}" if val is not None else " N/A "
        print(f"  {name:<30} {vs}  target≥{T[key]}  {pf(val,T[key])}")
    print(sep)
    print("\n  RECALL@5 BY TOPIC")
    for topic in df["topic_category"].unique():
        sub = df[df["topic_category"] == topic]
        print(f"  {topic:<28} {sub['hit_at_5'].mean():.3f}  (n={len(sub)})")
    print(sep + "\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="hybrid",
                    choices=["bm25_only","dense_only","hybrid","hybrid_no_faq"])
    ap.add_argument("--max-queries", type=int, default=None,
                    help="Take the first N queries from the full list")
    ap.add_argument("--per-doc", type=int, default=None,
                    help="Balanced sample: take N queries from each document (e.g. --per-doc 4 → 20 queries across 5 docs)")
    ap.add_argument("--no-ragas", action="store_true")
    args = ap.parse_args()
    run_evaluation(mode=args.mode, max_queries=args.max_queries,
                   per_doc=args.per_doc, use_ragas=not args.no_ragas)
