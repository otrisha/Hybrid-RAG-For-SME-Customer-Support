"""
generation/prompt_builder.py
=============================
Constructs the structured prompt passed to OpenAI.
The system prompt enforces grounding, citation, and graceful fallback.

Thesis reference: Sections 6.2 & 6.3.
"""

from __future__ import annotations
from retrieval.hybrid_retriever import RetrievedChunk

SYSTEM_PROMPT_TEMPLATE = """\
You are a technical customer support assistant for Benamdaj Dredging Solutions Ltd., \
a marine engineering company based in Nigeria. You assist customers with questions \
about four dredger models:

  - Model 1: 16/14 Inch Cutter Suction Dredger       (BDS-CSD-1614)
  - Model 2: 14/12 Inch Amphibious Multifunctional    (BDS-AMD-1412)
  - Model 3: 12/10 Inch Bucket Chain Dredger          (BDS-BCD-1210)
  - Model 4: 10/10 Inch Jet Suction Dredger           (BDS-JSD-1010)

INSTRUCTIONS — follow these strictly:

1. BASE YOUR ANSWER ONLY on the retrieved context passages provided below. \
Do not use any knowledge from outside the provided passages.

2. CITE YOUR SOURCES. After every factual claim, include the document reference \
in square brackets, e.g. [BDS-SPEC-001], [BDS-OM-001], [BDS-TSG-001], [BDS-FAQ-001].

3. IF THE CONTEXT DOES NOT CONTAIN SUFFICIENT INFORMATION to answer the question \
accurately, respond with exactly: "I do not have that information in the current \
documentation. Please contact Benamdaj Dredging Solutions Ltd. directly." \
Do NOT fabricate, guess, or extrapolate.

4. MODEL SPECIFICITY. If the customer specifies a model, restrict your answer \
to that model. State differences explicitly when information varies between models.

5. SAFETY QUERIES. For safety or emergency queries, always recommend following \
the Operations and Maintenance Manual (BDS-OM-001) and defer to qualified engineers.

6. TONE. Be clear, professional, and appropriately technical.

{safety_addendum}
---
RETRIEVED CONTEXT PASSAGES:

{context_passages}
---
"""

SAFETY_ADDENDUM = (
    "IMPORTANT — SAFETY-CRITICAL QUERY DETECTED: Provide accurate documentation "
    "information and strongly recommend direct contact with a qualified Benamdaj "
    "engineer before taking any action."
)


def format_context_passages(retrieved: list[RetrievedChunk]) -> str:
    if not retrieved:
        return "[No relevant passages retrieved from the knowledge base.]"
    blocks = []
    for i, rc in enumerate(retrieved, start=1):
        c = rc.chunk
        header = (f"[Passage {i} | Source: {c.document_id} | "
                  f"Section: {c.heading[:60]} | Model: {c.model}]")
        blocks.append(f"{header}\n{c.text.strip()}")
    return "\n\n---\n\n".join(blocks)


def build_prompt(query: str, retrieved: list[RetrievedChunk],
                 is_safety: bool = False) -> tuple[str, str]:
    """Return (system_prompt, user_message) for the OpenAI API call."""
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        safety_addendum  = SAFETY_ADDENDUM if is_safety else "",
        context_passages = format_context_passages(retrieved),
    )
    return system_prompt, query
