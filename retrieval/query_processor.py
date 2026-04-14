"""
retrieval/query_processor.py
=============================
Pre-processes incoming customer queries before retrieval:
  1. Cleans and normalises the query text
  2. Detects which dredger model the query concerns (if any)
  3. Classifies the query into a topic category
  4. Produces a metadata filter for Pinecone if a model is identified

Thesis reference: Section 5.5 (Query Pre-Processing and Model Routing).
"""

from __future__ import annotations
import re
from dataclasses import dataclass

from config.settings import MODEL_VOCABULARY
from utils.helpers import clean_text
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ProcessedQuery:
    original_query  : str
    cleaned_query   : str
    detected_model  : str | None
    topic_category  : str
    pinecone_filter : dict | None
    is_safety_query : bool = False
    is_multi_model  : bool = False
    is_greeting     : bool = False


_GREETING_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|good\s*(morning|afternoon|evening|day)|howdy|greetings|"
    r"what'?s\s*up|sup|how\s*are\s*you|how\s*do\s*you\s*do|nice\s*to\s*meet\s*you|"
    r"thanks|thank\s*you|cheers|bye|goodbye|see\s*you)\W*$",
    re.I,
)

_SAFETY_PATTERNS = [
    re.compile(r"fire|flood|man overboard|emergency|evacuate|spill|hazard", re.I),
    re.compile(r"safe(ty)?|danger|risk|ppe|life jacket|accident", re.I),
    re.compile(r"oil pressure.*low|engine.*knock|overheat.*alarm", re.I),
]

_MULTI_MODEL_PATTERNS = [
    re.compile(r"\bwhich\b.*(dredger|model|pump|engine)", re.I),
    re.compile(r"\bcompare\b|\bdifference\b|\bbest\b.*(model|dredger|for)", re.I),
    re.compile(r"(16.?14|14.?12|12.?10|10.?10).*(16.?14|14.?12|12.?10|10.?10)", re.I),
]

# Price/commercial queries should never be filtered by model — prices apply
# across the fleet and the price list is not split by model in the same way
_PRICE_PATTERN = re.compile(
    r"\b(price|cost|how much|fee|charge|rate|payment|discount|hire|rent|"
    r"purchase|buy|quote|invoice|warrant|deliver|export|import|budget)\b",
    re.I,
)

_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "equipment_knowledge" : ["what is","what does","how many","describe","specification",
                              "spec","capacity","hp","kilowatt","bore","stroke","dimension",
                              "how much does","what type"],
    "daily_operations"    : ["start","startup","how to start","shutdown","turn on","turn off",
                              "operate","warm up","anchor","winch","flush","shift"],
    "fault_diagnosis"     : ["not working","fault","problem","issue","broken","why is",
                              "won't","doesn't","error","smoke","noise","vibrat","leak",
                              "cavitat","pressure drop","no output","stall","stalling",
                              "keeps","overheating","knocking","alarm","warning"],
    "maintenance_practice": ["maintain","maintenance","service","oil change","filter","grease",
                              "how often","interval","schedule","200 hour","inspect"],
    "performance"         : ["production","output","rate","m3","slow","not producing",
                              "efficiency","capacity"],
    "safety"              : ["safe","danger","fire","flood","emergency","accident",
                              "ppe","life jacket","man overboard"],
    "spare_parts"         : ["spare","part","stock","order","price","impeller",
                              "packing","filter","buy"],
    "customer_interactions": ["warranty","support","contact","delivery","price","quote","purchase"],
}


def _classify_topic(query: str) -> str:
    q_lower = query.lower()
    scores  = {cat: sum(1 for kw in kws if kw in q_lower)
               for cat, kws in _TOPIC_KEYWORDS.items()}
    best    = max(scores, key=scores.get)
    return best if scores[best] > 0 else "equipment_knowledge"


def _detect_model(query: str) -> str | None:
    q_lower  = query.lower()
    detected = set()
    for model_label, keywords in MODEL_VOCABULARY.items():
        if any(kw.lower() in q_lower for kw in keywords):
            detected.add(model_label)
    return detected.pop() if len(detected) == 1 else None


def _build_pinecone_filter(model: str | None) -> dict | None:
    if model is None:
        return None
    return {"$or": [{"model": {"$eq": model}}, {"model": {"$eq": "All"}}]}


def process_query(query: str) -> ProcessedQuery:
    cleaned     = clean_text(query.strip())
    is_greeting = bool(_GREETING_PATTERNS.match(cleaned))
    is_multi    = any(p.search(cleaned) for p in _MULTI_MODEL_PATTERNS)
    is_price    = bool(_PRICE_PATTERN.search(cleaned))
    # Suppress model filter for greetings, multi-model, and price/commercial queries
    model       = None if (is_multi or is_greeting or is_price) else _detect_model(cleaned)
    topic       = _classify_topic(cleaned)
    is_safety   = any(p.search(cleaned) for p in _SAFETY_PATTERNS)

    log.debug(f"Query processed: model={model!r} | topic={topic} | safety={is_safety} | greeting={is_greeting} | price={is_price}")
    return ProcessedQuery(
        original_query  = query,
        cleaned_query   = cleaned,
        detected_model  = model,
        topic_category  = topic,
        pinecone_filter = _build_pinecone_filter(model),
        is_safety_query = is_safety,
        is_multi_model  = is_multi,
        is_greeting     = is_greeting,
    )
