"""
streamlit_app.py
================
Benamdaj Dredging Solutions Ltd — Hybrid RAG Customer Support
Streamlit Cloud deployment entry point.

Run locally:
    streamlit run streamlit_app.py

Thesis reference: Section 7 (Deployment).
"""

from __future__ import annotations
import os

import streamlit as st

# ── Inject Streamlit secrets into os.environ BEFORE any project imports ───────
# On Streamlit Cloud, API keys live in st.secrets (set via the dashboard).
# config/settings.py reads them via os.getenv(), so we bridge the two here.
def _apply_secrets() -> None:
    try:
        for key, val in st.secrets.items():
            if isinstance(val, str):
                os.environ.setdefault(key, val)
    except Exception:
        pass  # running locally — .env file handles this

_apply_secrets()

# ── Page config (must come before any other st.* calls) ──────────────────────
st.set_page_config(
    page_title="Benamdaj Dredging — Technical Support",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Project imports (after secrets are in env) ────────────────────────────────
from config.settings import BM25_INDEX_PATH, RETRIEVAL_MODE, validate_config
from retrieval.query_processor import process_query
from generation.generator import generate

# ── Resource loading (shared across all sessions, runs once per cold start) ───

@st.cache_resource(show_spinner="Loading knowledge base and embedding model…")
def _load_resources():
    """
    Load BM25 index and embedding model into memory.
    Triggers ingestion if the BM25 pickle is missing (should not happen in
    deployment since data/bm25_index.pkl is committed to the repository).
    """
    from ingestion.indexer import BM25Index, EmbeddingModel

    if not BM25_INDEX_PATH.exists():
        st.warning(
            "Knowledge base not found — running ingestion pipeline. "
            "This may take several minutes on first run…"
        )
        from ingest import run_ingest
        run_ingest()

    bm25 = BM25Index.load()
    EmbeddingModel.get()
    return bm25


@st.cache_resource(show_spinner=False)
def _get_retriever(mode: str):
    """Return a cached HybridRetriever for the given mode (one per mode string)."""
    from retrieval.hybrid_retriever import HybridRetriever
    bm25 = _load_resources()
    return HybridRetriever(bm25, mode=mode)


# ── UI helpers ────────────────────────────────────────────────────────────────

def _render_sources(sources: list[str]) -> None:
    if not sources:
        return
    tags = "  ".join(f"`{s}`" for s in sources)
    st.caption(f"Sources: {tags}")


def _render_metadata(meta: dict) -> None:
    topic   = meta.get("topic", "")
    mode    = meta.get("mode", "")
    latency = meta.get("latency", 0.0)
    dmodel  = meta.get("dredger_model") or "—"
    cols    = st.columns(4)
    cols[0].caption(f"Topic: **{topic}**")
    cols[1].caption(f"Mode: **{mode}**")
    cols[2].caption(f"Model: **{dmodel}**")
    cols[3].caption(f"⏱ {latency:.2f} s")


def _render_assistant_body(msg: dict) -> None:
    st.markdown(msg["content"])
    if msg.get("is_fallback"):
        st.warning(
            "I do not have specific information on that in the knowledge base. "
            "Please contact Benamdaj Dredging Solutions Ltd. directly."
        )
    _render_sources(msg.get("sources", []))
    _render_metadata(msg.get("meta", {}))


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _sidebar() -> str:
    with st.sidebar:
        st.markdown("## ⚓ Benamdaj Dredging")
        st.markdown("**Technical Support Assistant**")
        st.divider()

        mode = st.selectbox(
            "Retrieval Mode",
            options=["hybrid", "bm25_only", "dense_only", "hybrid_no_faq"],
            index=0,
            help=(
                "**hybrid** — BM25 + Pinecone with RRF fusion (recommended)\n\n"
                "**bm25_only** — keyword search only\n\n"
                "**dense_only** — semantic search only\n\n"
                "**hybrid_no_faq** — hybrid without FAQ corpus"
            ),
        )

        st.divider()

        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption(
            "BDS-RAG-001\n\n"
            "MSc AI & Data Science\n\n"
            "University of Wolverhampton\n\n"
            "Patricia Orji"
        )

    return mode


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    if not validate_config():
        st.error(
            "API keys are not configured. "
            "Add OPENAI_API_KEY and PINECONE_API_KEY to your Streamlit secrets or .env file."
        )
        st.stop()

    mode = _sidebar()

    st.title("⚓ Benamdaj Dredging Solutions")
    st.caption(
        "Ask anything about dredger specifications, operating procedures, "
        "troubleshooting, maintenance schedules, or pricing."
    )

    # Session state initialisation
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Load retriever for current mode
    retriever = _get_retriever(mode)

    # Replay conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                _render_assistant_body(msg)
            else:
                st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask about Benamdaj dredgers…"):

        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating…"):
                pq        = process_query(prompt)
                retrieved = retriever.retrieve(pq)
                response  = generate(pq, retrieved, mode=mode)

            assistant_msg = {
                "role"    : "assistant",
                "content" : response.answer,
                "sources" : response.sources_cited,
                "is_fallback": response.is_fallback,
                "meta"    : {
                    "topic"         : response.topic_category,
                    "mode"          : response.retrieval_mode,
                    "latency"       : response.latency_seconds,
                    "dredger_model" : response.detected_model,
                },
            }
            _render_assistant_body(assistant_msg)

        st.session_state.messages.append(assistant_msg)


main()
