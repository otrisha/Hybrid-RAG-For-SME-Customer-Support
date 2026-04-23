# Benamdaj Hybrid RAG — Customer Support System
### BDS-RAG-001 | MSc AI & Data Science | University of Wolverhampton

**Researcher:** Patricia Orji  
**Supervisor:** [Supervisor Name]  
**Dissertation title:** *Hybrid Retrieval-Augmented Generation (RAG) for SME Customer Support: A Case Study in Dredging Equipment Manufacturing*

---

## Overview

This codebase implements a Hybrid Retrieval-Augmented Generation (RAG) system for **Benamdaj Dredging Solutions Ltd.** (RC: 1653951), a marine engineering SME based in Nigeria. The system replaces a human-staffed live chat — unavailable outside working hours, inconsistent across agents, and unable to handle concurrent queries — with an AI-powered assistant grounded in the company's own technical documentation.

The system retrieves relevant content from a five-document knowledge base using a combination of **BM25 sparse retrieval** and **dense semantic retrieval** (Pinecone), fuses results via **Reciprocal Rank Fusion (RRF)**, and generates grounded, cited responses using the **OpenAI API**. It is deployed as a **Streamlit** web application on Streamlit Community Cloud.

---

## Live Demo

> Deployed on Streamlit Community Cloud  
> [View on](https://benamdajtechdesk.streamlit.app/)

---

## Project Structure

```
Thesis/
├── .streamlit/
│   ├── config.toml               Streamlit theme (navy/blue)
│   └── secrets.toml.example      Template — copy and fill for local Streamlit dev
│
├── config/
│   ├── settings.py               All constants, API keys, paths, evaluation targets
│   └── .env                      Local environment variables (not committed)
│
├── data/
│   ├── documents/                Five source PDFs (committed to repo)
│   ├── chunks/                   Chunked JSON output from ingestion
│   └── bm25_index.pkl            Pre-built BM25 index (committed — avoids cold-start re-ingestion)
│
├── ingestion/
│   ├── document_loader.py        PDF parsing via pdfplumber; font-size heading inference
│   ├── chunker.py                Four domain-adaptive chunking strategies
│   └── indexer.py                BM25Index, EmbeddingModel, Pinecone upsert
│
├── retrieval/
│   ├── query_processor.py        Query cleaning, model detection, topic classification, greeting handling
│   └── hybrid_retriever.py       BM25 + dense + RRF fusion; four ablation modes
│
├── generation/
│   ├── prompt_builder.py         System prompt construction with context formatting
│   └── generator.py              OpenAI API call, citation verification, fallback detection
│
├── evaluation/
│   ├── eval_queries.py           93-query evaluation set with ground truths (5 documents)
│   └── ragas_evaluator.py        Recall@5, Precision@5, MRR + RAGAS metrics; ablation runner
│
├── utils/
│   ├── helpers.py                Text cleaning, tokenisation, chunk ID generation
│   └── logger.py                 Structured logging
│
├── docs/
│   ├── generate_chapter4.py      Generates Chapter 4 (Methodology) .docx
│   └── generate_chapter5.py      Generates Chapter 5 (Results) .docx
│
├── streamlit_app.py              Streamlit web application (deployment entry point)
├── main.py                       Flask web app + CLI (local development)
├── ingest.py                     Knowledge base construction entry point
├── requirements.txt              Deployment requirements (Streamlit Cloud)
└── requirements_eval.txt         Evaluation requirements (local only)
```

---

## Knowledge Base

| Document ID | File | Knowledge Type | Chunking Strategy | Chunks |
|---|---|---|---|---|
| BDS-SPEC-001 | `Benamdaj_Product_Specification_Manual.pdf` | Explicit — structured | Section-based | ~21 |
| BDS-OM-001 | `Benamdaj_OM_Manual.pdf` | Explicit — procedural | Procedure-based | ~19 |
| BDS-TSG-001 | `Benamdaj_Troubleshooting_Guide.pdf` | Explicit — diagnostic | Fault-block-based | ~14 |
| BDS-FAQ-001 | `BDS-FAQ-001_Staff_Interview_FAQs.pdf` | Tacit — elicited | Q&A-pair-based | 56 |
| BDS-PL-001 | `Benamdaj_price_list.pdf` | Explicit — structured | Section-based | ~19 |
| **Total** | | | | **109 chunks** |

---

## Quick Start — Local Development

### 1. Clone and install

```bash
git clone https://github.com/otrisha/Hybrid-RAG-For-SME-Customer-Support.git
cd Hybrid-RAG-For-SME-Customer-Support

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements_eval.txt   # includes all deployment + evaluation deps
```

### 2. Configure environment

Create `config/.env`:

```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=benamdaj-rag
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
OPENAI_MODEL=gpt-4o-mini
```

### 3. Run the Streamlit app (recommended)

```bash
streamlit run streamlit_app.py
```

The knowledge base is loaded automatically on first run. If `data/bm25_index.pkl` is present (it is, committed to the repo) and Pinecone is already populated, the app starts in seconds.

### 4. Or run the Flask app (alternative local server)

```bash
python main.py                 # Flask on http://localhost:5000
python main.py --cli           # Interactive CLI mode
python main.py --mode bm25_only
```

### 5. Re-ingest from scratch (if needed)

```bash
python ingest.py
```

This re-chunks all PDFs, rebuilds the BM25 index, and re-uploads vectors to Pinecone.

---

## Streamlit Cloud Deployment

1. Push this repository to GitHub (including `data/bm25_index.pkl` and all PDFs)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select the repository and set **Main file path** to `streamlit_app.py`
4. Under **Settings → Secrets**, paste the contents of `.streamlit/secrets.toml.example` filled with real values
5. Deploy — first cold start takes ~60 seconds to download the sentence-transformers model; subsequent starts are fast

---

## Retrieval Modes (Ablation Study)

| Mode | Description | Streamlit selector |
|---|---|---|
| `hybrid` | BM25 + Dense + RRF — full proposed system (default) | ✓ |
| `bm25_only` | Sparse keyword retrieval only (Config A baseline) | ✓ |
| `dense_only` | Dense semantic retrieval only (Config B baseline) | ✓ |
| `hybrid_no_faq` | Hybrid without BDS-FAQ-001 tacit knowledge (Config D) | ✓ |

Switch modes via the sidebar dropdown in the Streamlit app, or via `--mode` flag in the Flask/CLI app.

---

## Running the Evaluation

Install evaluation dependencies first:

```bash
pip install -r requirements_eval.txt
```

```bash
# Balanced 20-query run with RAGAS (4 queries per document, ~$0.10)
python -m evaluation.ragas_evaluator --mode hybrid --per-doc 4

# Full 93-query run with RAGAS
python -m evaluation.ragas_evaluator --mode hybrid

# Retrieval metrics only — no RAGAS, no API cost
python -m evaluation.ragas_evaluator --mode hybrid --no-ragas

# Ablation study — all four configurations
for mode in bm25_only dense_only hybrid hybrid_no_faq; do
    python -m evaluation.ragas_evaluator --mode $mode --per-doc 4
done
```

Results are saved as CSV files in `logs/evaluation/eval_<mode>_<timestamp>.csv`.

---

## Evaluation Results (Hybrid — Configuration C)

Evaluation run: 20 queries, balanced across 5 documents.

| Metric | Result | Target | Status |
|---|---|---|---|
| Recall@5 | 0.800 | ≥ 0.80 | PASS |
| Precision@5 | 0.390 | ≥ 0.40 | FAIL |
| MRR | 0.685 | ≥ 0.70 | FAIL |
| Fallback rate | 0.000 | — | — |
| Citation rate | 1.000 | — | — |
| Mean latency | 3.47 s | ≤ 8.0 s | PASS |
| RAGAS Faithfulness | pending | ≥ 0.80 | — |
| RAGAS Answer Relevancy | pending | ≥ 0.75 | — |
| RAGAS Context Precision | pending | ≥ 0.65 | — |
| RAGAS Context Recall | pending | ≥ 0.75 | — |

> Precision@5 and MRR failures are discussed in Chapter 5. With a 5-document corpus, cross-document retrieval is expected; Precision@5 = 0.39 is 1.95× the random baseline of 0.20.

---

## Evaluation Targets

| Metric | Target | Rationale |
|---|---|---|
| Recall@5 | ≥ 0.80 | Primary retrieval success criterion |
| Precision@5 | ≥ 0.40 | Adjusted for 5-doc corpus (2× random baseline) |
| MRR | ≥ 0.70 | First-rank relevance |
| RAGAS Faithfulness | ≥ 0.80 | Hallucination guard |
| RAGAS Answer Relevancy | ≥ 0.75 | Response usefulness |
| RAGAS Context Precision | ≥ 0.65 | Retrieval precision proxy |
| RAGAS Context Recall | ≥ 0.75 | Retrieval coverage |
| Max Latency | ≤ 8.0 s | Practical usability |

---

## Technology Stack

| Component | Technology |
|---|---|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Vector database | Pinecone Serverless (AWS us-east-1, cosine similarity) |
| Sparse retrieval | BM25Okapi — `rank-bm25==0.2.2` (k₁=1.5, b=0.75) |
| Fusion | Reciprocal Rank Fusion (RRF, k=60) |
| Generator | OpenAI `gpt-4o-mini` |
| PDF parsing | `pdfplumber` with font-size heading inference |
| Web UI | Streamlit |
| Evaluation | RAGAS 0.4.3 |

---

## References

- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *arXiv:2005.11401*
- Gao et al. (2024). RAG for Large Language Models: A Survey. *arXiv:2312.10997*
- Robertson & Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *FnTIR 3(4)*
- Sawarkar et al. (2024). Blended RAG. *arXiv:2404.07220*
- Es et al. (2023). RAGAS: Automated Evaluation of RAG. *arXiv:2309.15217*
