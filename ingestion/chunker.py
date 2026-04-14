"""
ingestion/chunker.py
====================
Domain-adaptive chunking — four strategies matched to each document type.

  BDS-SPEC-001  →  section_chunker()     (section-based)
  BDS-OM-001    →  procedure_chunker()   (procedure-based)
  BDS-TSG-001   →  fault_block_chunker() (fault-block-based)
  BDS-FAQ-001   →  qa_pair_chunker()     (Q&A-pair-based)

Thesis reference: Section 4.2 (Chunking Strategy).
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Callable

from config.settings import MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS
from ingestion.document_loader import LoadedDocument, DocumentBlock
from utils.helpers import count_tokens, generate_chunk_id, clean_text
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Chunk:
    chunk_id          : str
    document_id       : str
    document_title    : str
    source            : str
    knowledge_type    : str
    chunking_method   : str
    text              : str
    bm25_text         : str
    heading           : str = ""
    model             : str = "All"
    topic_category    : str = ""
    persona_role      : str = ""
    retrieval_priority: str = "medium"
    token_count       : int = 0

    def __post_init__(self):
        self.token_count = count_tokens(self.text)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_model_from_text(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["16/14", "16 inch", "model 1", "csd-1614", "cutter suction dredger"]):
        return "Model 1"
    if any(k in t for k in ["14/12", "14 inch", "model 2", "amd-1412", "amphibious"]):
        return "Model 2"
    if any(k in t for k in ["12/10", "12 inch", "model 3", "bcd-1210", "bucket chain", "cat 3408", "3408"]):
        return "Model 3"
    if any(k in t for k in ["10/10", "10 inch", "model 4", "jsd-1010", "jet suction", "wp13"]):
        return "Model 4"
    return "All"


def _group_by_heading(blocks: list[DocumentBlock],
                       level: int = 2) -> list[tuple[str, list[DocumentBlock]]]:
    sections: list[tuple[str, list[DocumentBlock]]] = []
    heading  = "Introduction"
    current  : list[DocumentBlock] = []
    for block in blocks:
        if block.block_type == "heading" and block.level <= level:
            if current:
                sections.append((heading, current))
            heading = block.text
            current = []
        else:
            if block.text.strip():
                current.append(block)
    if current:
        sections.append((heading, current))
    return sections


def _merge_short(chunks: list[Chunk]) -> list[Chunk]:
    if not chunks:
        return chunks
    merged = [chunks[0]]
    for c in chunks[1:]:
        if c.token_count < MIN_CHUNK_TOKENS:
            prev = merged[-1]
            combined = prev.text + "\n\n" + c.text
            prev.text = combined
            prev.bm25_text = combined
            prev.token_count = count_tokens(combined)
        else:
            merged.append(c)
    return merged


# ── Strategy 1: Section-based (BDS-SPEC-001) ──────────────────────────────────

def section_chunker(doc: LoadedDocument) -> list[Chunk]:
    log.info(f"  section_chunker → {doc.document_id}")
    chunks: list[Chunk] = []
    sections = _group_by_heading(doc.blocks, level=2)

    for idx, (heading, blocks) in enumerate(sections):
        text = "\n\n".join(b.text for b in blocks if b.text.strip())
        if not text.strip():
            continue
        model = _detect_model_from_text(heading + " " + text)

        if count_tokens(text) > MAX_CHUNK_TOKENS:
            sub_sections = _group_by_heading(blocks, level=3)
            for sidx, (sub_h, sub_b) in enumerate(sub_sections):
                sub_text = "\n\n".join(b.text for b in sub_b if b.text.strip())
                if not sub_text.strip():
                    continue
                cid = generate_chunk_id(doc.document_id, idx*100+sidx, f"{heading}-{sub_h}")
                chunks.append(Chunk(
                    chunk_id=cid, document_id=doc.document_id, document_title=doc.title,
                    source=doc.source, knowledge_type=doc.knowledge_type,
                    chunking_method=doc.chunking_method,
                    text=f"{heading} — {sub_h}\n\n{sub_text}",
                    bm25_text=f"{heading} {sub_h} {sub_text}",
                    heading=f"{heading} — {sub_h}", model=model, retrieval_priority="high",
                ))
        else:
            cid = generate_chunk_id(doc.document_id, idx, heading)
            chunks.append(Chunk(
                chunk_id=cid, document_id=doc.document_id, document_title=doc.title,
                source=doc.source, knowledge_type=doc.knowledge_type,
                chunking_method=doc.chunking_method,
                text=f"{heading}\n\n{text}", bm25_text=f"{heading} {text}",
                heading=heading, model=model, retrieval_priority="high",
            ))

    chunks = _merge_short(chunks)
    log.info(f"  {doc.document_id}: {len(chunks)} section chunks")
    return chunks


# ── Strategy 2: Procedure-based (BDS-OM-001) ──────────────────────────────────

_PROC_RE = [
    re.compile(r"pre.?start|startup|start.up|shutdown|shut.down|pump.*engag|"
               r"post.?shutdown|daily.*maint|weekly.*maint|200.?hour|"
               r"quarterly|annual|emergency|fire|flood|man overboard|"
               r"lubrication|grease.*chart|operating.*guideline|limit", re.I)
]

def _is_proc_heading(text: str) -> bool:
    return any(p.search(text) for p in _PROC_RE)

def procedure_chunker(doc: LoadedDocument) -> list[Chunk]:
    log.info(f"  procedure_chunker → {doc.document_id}")
    chunks: list[Chunk] = []
    sections = _group_by_heading(doc.blocks, level=2)

    for sidx, (heading, blocks) in enumerate(sections):
        cur_h = heading
        cur_b: list[DocumentBlock] = []
        pidx  = 0

        for block in blocks:
            if (block.block_type in ("heading", "paragraph") and
                    _is_proc_heading(block.text) and block.level >= 2):
                if cur_b:
                    text = "\n\n".join(b.text for b in cur_b)
                    if text.strip():
                        cid = generate_chunk_id(doc.document_id, sidx*100+pidx, cur_h)
                        chunks.append(Chunk(
                            chunk_id=cid, document_id=doc.document_id, document_title=doc.title,
                            source=doc.source, knowledge_type=doc.knowledge_type,
                            chunking_method=doc.chunking_method,
                            text=f"{cur_h}\n\n{text}", bm25_text=f"{cur_h} {text}",
                            heading=cur_h, model="All", retrieval_priority="high",
                        ))
                        pidx += 1
                cur_h = block.text
                cur_b = []
            else:
                if block.text.strip():
                    cur_b.append(block)

        if cur_b:
            text = "\n\n".join(b.text for b in cur_b)
            if text.strip():
                cid = generate_chunk_id(doc.document_id, sidx*100+pidx, cur_h)
                chunks.append(Chunk(
                    chunk_id=cid, document_id=doc.document_id, document_title=doc.title,
                    source=doc.source, knowledge_type=doc.knowledge_type,
                    chunking_method=doc.chunking_method,
                    text=f"{cur_h}\n\n{text}", bm25_text=f"{cur_h} {text}",
                    heading=cur_h, model="All", retrieval_priority="high",
                ))

    if not chunks:
        log.warning(f"  No procedures detected in {doc.document_id}; falling back to section_chunker")
        return section_chunker(doc)
    chunks = _merge_short(chunks)
    log.info(f"  {doc.document_id}: {len(chunks)} procedure chunks")
    return chunks


# ── Strategy 3: Fault-block-based (BDS-TSG-001) ───────────────────────────────

def fault_block_chunker(doc: LoadedDocument) -> list[Chunk]:
    log.info(f"  fault_block_chunker → {doc.document_id}")
    chunks: list[Chunk] = []
    raw    = doc.raw_text
    parts  = re.split(r"(?=SYMPTOM\s*:)", raw, flags=re.I)

    # First part = introductory reference index
    intro = parts[0].strip()
    if intro:
        chunks.append(Chunk(
            chunk_id=generate_chunk_id(doc.document_id, 0, "symptom-reference-index"),
            document_id=doc.document_id, document_title=doc.title,
            source=doc.source, knowledge_type=doc.knowledge_type,
            chunking_method=doc.chunking_method,
            text=intro, bm25_text=intro,
            heading="Quick Symptom Reference Index",
            model="All", retrieval_priority="high",
        ))

    for idx, part in enumerate(parts[1:], start=1):
        part = part.strip()
        if not part:
            continue
        first_line   = part.split("\n")[0].strip()
        symptom_text = re.sub(r"^SYMPTOM\s*:\s*", "", first_line, flags=re.I).strip()
        model        = _detect_model_from_text(part)
        chunks.append(Chunk(
            chunk_id=generate_chunk_id(doc.document_id, idx, symptom_text),
            document_id=doc.document_id, document_title=doc.title,
            source=doc.source, knowledge_type=doc.knowledge_type,
            chunking_method=doc.chunking_method,
            text=part, bm25_text=part,
            heading=f"SYMPTOM: {symptom_text}",
            model=model, topic_category="fault_diagnosis", retrieval_priority="high",
        ))

    log.info(f"  {doc.document_id}: {len(chunks)} fault-block chunks")
    return chunks


# ── Strategy 4: Q&A-pair-based (BDS-FAQ-001) ──────────────────────────────────

_FAQ_CATEGORY_KEYWORDS = {
    "equipment_knowledge" : ["model","dredger","engine","pump","hp","kilowatt",
                              "inch","depth","discharge","jet","cutter"],
    "daily_operations"    : ["start","startup","shutdown","warm","idle","anchor",
                              "winch","flush","shift","rpm","operate"],
    "fault_diagnosis"     : ["fault","fail","symptom","diagnos","pressure","cavitat",
                              "smoke","knock","overheat","leak","vibrat","stall","block"],
    "maintenance_practice": ["maint","oil","filter","grease","service","interval",
                              "200 hour","lubric","replace","change"],
    "performance"         : ["production","output","m3","cubic","slow","capacity",
                              "afternoon","density","rate"],
    "safety"              : ["safe","fire","flood","man overboard","life","ppe",
                              "jacket","evacuate","emergency","weather"],
    "spare_parts"         : ["spare","part","stock","order","impeller",
                              "packing","filter","wire","battery"],
    "customer_interactions": ["client","customer","ask","question","warrant",
                               "price","fuel consumption","24 hour","hire"],
}

_PERSONA_MAP = {
    "S1 — Senior Operator"      : ["S1","Senior Operator","Senior Dredge Operator"],
    "S2 — Junior Operator"      : ["S2","Junior Operator","Junior Dredge Operator"],
    "S3 — Chief Technician"     : ["S3","Chief","Chief Maintenance","Chief Technician"],
    "S4 — Field Technician"     : ["S4","Field Service","Field Technician"],
    "S5 — Operations Supervisor": ["S5","Supervisor","Operations Supervisor"],
    "S6 — Workshop Technician"  : ["S6","Workshop","Workshop Technician"],
}

def _classify_faq_topic(text: str) -> str:
    tl     = text.lower()
    scores = {cat: sum(1 for kw in kws if kw in tl)
              for cat, kws in _FAQ_CATEGORY_KEYWORDS.items()}
    best   = max(scores, key=scores.get)
    return best if scores[best] > 0 else "equipment_knowledge"

def _extract_persona(ctx: str) -> str:
    for label, patterns in _PERSONA_MAP.items():
        if any(p in ctx for p in patterns):
            return label
    return "Unknown"

def qa_pair_chunker(doc: LoadedDocument) -> list[Chunk]:
    log.info(f"  qa_pair_chunker → {doc.document_id}")
    chunks: list[Chunk] = []
    raw    = doc.raw_text

    qa_pattern = re.compile(
        r"Q\s*\n(.+?)\n\s*A\s*\n(.+?)(?=\nQ\s*\n|\Z)",
        re.DOTALL | re.IGNORECASE
    )
    matches = list(qa_pattern.finditer(raw))

    if not matches:
        log.warning(f"  Standard Q/A pattern not found in {doc.document_id}; falling back to section_chunker")
        return section_chunker(doc)

    for idx, match in enumerate(matches, start=1):
        q_text   = clean_text(match.group(1))
        a_text   = clean_text(match.group(2))
        full     = f"Q: {q_text}\n\nA: {a_text}"
        topic    = _classify_faq_topic(full)
        persona  = _extract_persona(raw[max(0, match.start()-300):match.start()])
        model    = _detect_model_from_text(full)
        bm25_txt = f"{q_text} {topic.replace('_',' ')}"

        chunks.append(Chunk(
            chunk_id=generate_chunk_id(doc.document_id, idx, q_text[:40]),
            document_id=doc.document_id, document_title=doc.title,
            source=doc.source, knowledge_type=doc.knowledge_type,
            chunking_method=doc.chunking_method,
            text=full, bm25_text=bm25_txt,
            heading=f"FAQ-{idx:02d}: {q_text[:60]}",
            model=model, topic_category=topic, persona_role=persona,
            retrieval_priority="high" if topic in (
                "fault_diagnosis","daily_operations","maintenance_practice") else "medium",
        ))

    log.info(f"  {doc.document_id}: {len(chunks)} Q&A-pair chunks")
    return chunks


# ── Dispatch ──────────────────────────────────────────────────────────────────

_STRATEGY: dict[str, Callable] = {
    "section"    : section_chunker,
    "procedure"  : procedure_chunker,
    "fault_block": fault_block_chunker,
    "qa_pair"    : qa_pair_chunker,
}

def chunk_document(doc: LoadedDocument) -> list[Chunk]:
    fn = _STRATEGY.get(doc.chunking_method)
    if fn is None:
        raise ValueError(f"Unknown chunking method: {doc.chunking_method}")
    return fn(doc)

def chunk_all_documents(docs: list[LoadedDocument]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    log.info(f"Total chunks: {len(all_chunks)}")
    return all_chunks

# Alias for test compatibility
_group_blocks_by_heading = _group_by_heading
