# Benamdaj Hybrid RAG — Customer Support System
### BDS-RAG-001 | MSc AI & Data Science | University of Wolverhampton

**Researcher:** Patricia Orji  
**Supervisor:** [Supervisor Name]  
**Dissertation title:** *Hybrid Retrieval-Augmented Generation (RAG) for SME Customer Support: A Case Study in Dredging Equipment Manufacturing*

---

## Overview

This codebase implements a Hybrid Retrieval-Augmented Generation (RAG) system for Benamdaj Dredging Solutions Ltd. (RC: 1653951), a marine engineering SME based in Nigeria. The system replaces a human-staffed live chat — which is unavailable outside agent working hours, inconsistent across agents, and unable to handle concurrent queries — with an AI-powered assistant grounded in the company's own technical documentation.

The system retrieves relevant content from a four-document knowledge base using a combination of BM25 sparse retrieval and dense semantic retrieval, fuses results via Reciprocal Rank Fusion (RRF), and generates grounded, cited responses using the OpenAI API.

---

## Project Structure

```
benamdaj_rag/
├── config/
│   ├── settings.py           All constants, API keys, paths, evaluation targets
│   ├── prompts.py            System prompt templates
│   └── .env.example          Environment variable template
│
├── ingestion/
│   ├── document_loader.py    Parses .docx files; preserves heading structure
│   ├── chunker.py            Four domain-adaptive chunking strategies
│   └── indexer.py            BM25Index, EmbeddingModel, Pinecone upsert
│
├── retrieval/
│   ├── query_processor.py    Query cleaning, model detection, topic classification
│   └── hybrid_retriever.py   BM25 + dense + RRF fusion; ablation support
│
├── generation/
│   ├── prompt_builder.py     System prompt construction with context formatting
│   └── generator.py          OpenAI API call, citation verification, fallback detection
│
├── evaluation/
│   ├── eval_queries.py       100-query evaluation set with ground truths
│   └── ragas_evaluator.py    Recall@5, Precision@5, MRR + RAGAS metrics; ablation runner
│
├── utils/
│   ├── helpers.py            Text cleaning, tokenisation, chunk ID generation
│   └── logger.py             Coloured structured logging
│
├── tests/
│   └── test_pipeline.py      32 unit tests (no API keys required)
│
├── ingest.py                 Knowledge base construction entry point
├── main.py                   CLI + FastAPI API server
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp config/.env.example .env
# Edit .env and add your OPENAI_API_KEY and PINECONE_API_KEY
```

### 3. Place documents in the data directory
```
data/documents/
├── BDS-SPEC-001_Product_Specification_Manual.docx
├── BDS-OM-001_Operations_Maintenance_Manual.docx
├── BDS-TSG-001_Troubleshooting_Guide.docx
└── BDS-FAQ-001_Staff_Interview_FAQs.docx
```

### 4. Ingest the knowledge base
```bash
python ingest.py
# Chunks all documents, builds BM25 index, uploads to Pinecone
```

### 5. Run the interactive CLI
```bash
python main.py
# Starts interactive customer support chat session
```

### 6. Or run the FastAPI server
```bash
uvicorn main:app --reload
# API available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

---

## Ablation Configurations

The system supports four retrieval modes for the ablation study (Thesis Table 8):

| Mode | Description | Run command |
|------|-------------|-------------|
| `hybrid` | BM25 + Dense + RRF (proposed system, default) | `python main.py` |
| `bm25_only` | Sparse retrieval only (Config A baseline) | `python main.py --mode bm25_only` |
| `dense_only` | Dense retrieval only (Config B baseline) | `python main.py --mode dense_only` |
| `hybrid_no_faq` | Hybrid without BDS-FAQ-001 (Config D) | `python main.py --mode hybrid_no_faq` |

---

## Running the Evaluation

```bash
# Full evaluation with RAGAS (requires OpenAI key)
python -m evaluation.ragas_evaluator --mode hybrid

# Ablation — all four configurations
for mode in bm25_only dense_only hybrid hybrid_no_faq; do
    python -m evaluation.ragas_evaluator --mode $mode
done

# Quick retrieval-only evaluation (no RAGAS, no API key needed)
python -m evaluation.ragas_evaluator --mode hybrid --no-ragas
```

Results are saved as CSV files in `logs/evaluation/`.

---

## Running Unit Tests

```bash
# Run all 32 tests (no API keys required)
python -m pytest tests/test_pipeline.py -v
```

---

## Knowledge Base Documents

| Document ID | Title | Knowledge Type | Chunking |
|-------------|-------|---------------|----------|
| BDS-SPEC-001 | Product Specification Manual | Explicit — structured | Section-based (~50 chunks) |
| BDS-OM-001 | Operations & Maintenance Manual | Explicit — procedural | Procedure-based (~40 chunks) |
| BDS-TSG-001 | Troubleshooting Guide | Explicit — diagnostic | Fault-block-based (~30 chunks) |
| BDS-FAQ-001 | Staff Interview FAQs | Tacit — elicited | Q&A-pair-based (56 chunks) |

---

## Evaluation Targets (Thesis Tables 6 & 7)

| Metric | Target |
|--------|--------|
| Recall@5 | ≥ 0.80 |
| Precision@5 | ≥ 0.60 |
| MRR | ≥ 0.70 |
| RAGAS Faithfulness | ≥ 0.80 |
| RAGAS Answer Relevancy | ≥ 0.75 |
| RAGAS Context Precision | ≥ 0.65 |
| RAGAS Context Recall | ≥ 0.75 |
| Max Latency | ≤ 8.0s |

---

## References

- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *arXiv:2005.11401*
- Gao et al. (2024). RAG for Large Language Models: A Survey. *arXiv:2312.10997*
- Robertson & Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *FnTIR 3(4)*
- Sawarkar et al. (2024). Blended RAG. *arXiv:2404.07220*
- Es et al. (2023). RAGAS: Automated Evaluation of RAG. *arXiv:2309.15217*
