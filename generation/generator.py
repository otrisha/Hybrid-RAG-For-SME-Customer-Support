"""
generation/generator.py
========================
Calls the OpenAI API and post-processes the response.
Post-processing: verifies citations, detects fallback, logs misses.

Thesis reference: Sections 6.1 & 6.4.
"""

from __future__ import annotations
import re
import time
from dataclasses import dataclass, field

from openai import OpenAI

from config.settings import OPENAI_API_KEY, OPENAI_MODEL, MAX_TOKENS, TEMPERATURE
from generation.prompt_builder import build_prompt
from retrieval.hybrid_retriever import RetrievedChunk
from retrieval.query_processor import ProcessedQuery
from utils.logger import get_logger

log = get_logger(__name__)

_client: OpenAI | None = None
FALLBACK_PHRASE = "I do not have that information"
_CITATION_RE    = re.compile(r"\[BDS-(?:SPEC|OM|TSG|FAQ|PL)-\d+\]")

_GREETING_RESPONSES = {
    "hi"       : "Hello! Welcome to Benamdaj Dredging Solutions technical support. How can I help you today?",
    "hello"    : "Hello! Welcome to Benamdaj Dredging Solutions technical support. How can I help you today?",
    "hey"      : "Hey there! Welcome to Benamdaj technical support. What can I help you with?",
    "thanks"   : "You're welcome! Is there anything else I can help you with?",
    "thank you": "You're welcome! Feel free to ask if you have any other questions.",
    "bye"      : "Goodbye! Don't hesitate to reach out if you need further assistance.",
    "goodbye"  : "Goodbye! Don't hesitate to reach out if you need further assistance.",
    "default"  : "Hello! I'm the Benamdaj technical support assistant. I can help with equipment specifications, operating procedures, troubleshooting, maintenance, and pricing. What would you like to know?",
}


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


@dataclass
class RAGResponse:
    query            : str
    answer           : str
    retrieved_chunks : list[RetrievedChunk]
    sources_cited    : list[str]
    is_fallback      : bool
    has_citations    : bool
    model_used       : str
    latency_seconds  : float
    detected_model   : str | None
    topic_category   : str
    retrieval_mode   : str


def _extract_sources(answer: str) -> list[str]:
    found = _CITATION_RE.findall(answer)
    return list(dict.fromkeys(tag.strip("[]") for tag in found))


def _greeting_response(pq: ProcessedQuery) -> RAGResponse:
    """Return a friendly reply without touching retrieval or the OpenAI API."""
    key    = pq.cleaned_query.lower().rstrip("!.,?")
    answer = _GREETING_RESPONSES.get(key, _GREETING_RESPONSES["default"])
    return RAGResponse(
        query=pq.original_query, answer=answer, retrieved_chunks=[],
        sources_cited=[], is_fallback=False, has_citations=False,
        model_used="none", latency_seconds=0.0,
        detected_model=None, topic_category="general",
        retrieval_mode="none",
    )


def generate(pq: ProcessedQuery, retrieved: list[RetrievedChunk],
             mode: str = "hybrid") -> RAGResponse:
    if pq.is_greeting:
        return _greeting_response(pq)

    system_prompt, user_message = build_prompt(
        query=pq.cleaned_query, retrieved=retrieved, is_safety=pq.is_safety_query
    )
    t0 = time.perf_counter()
    try:
        resp = _get_client().chat.completions.create(
            model=OPENAI_MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as exc:
        log.error(f"OpenAI API call failed: {exc}")
        answer = ("I am unable to process your request at this time. "
                  "Please contact Benamdaj Dredging Solutions Ltd. directly.")

    latency       = time.perf_counter() - t0
    is_fallback   = FALLBACK_PHRASE.lower() in answer.lower()
    sources_cited = _extract_sources(answer)
    has_citations = bool(sources_cited)

    if not is_fallback and not has_citations:
        log.warning(f"No citations in response for: '{pq.cleaned_query[:80]}'")
    if is_fallback:
        log.info(f"Retrieval miss | query='{pq.cleaned_query[:60]}'")
    else:
        log.info(f"Response OK | {latency:.2f}s | sources={sources_cited} | model={pq.detected_model!r}")

    return RAGResponse(
        query=pq.original_query, answer=answer, retrieved_chunks=retrieved,
        sources_cited=sources_cited, is_fallback=is_fallback, has_citations=has_citations,
        model_used=OPENAI_MODEL, latency_seconds=latency,
        detected_model=pq.detected_model, topic_category=pq.topic_category,
        retrieval_mode=mode,
    )
