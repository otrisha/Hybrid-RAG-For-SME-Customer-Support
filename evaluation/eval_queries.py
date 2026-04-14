"""
evaluation/eval_queries.py
===========================
120-query evaluation set spanning all five documents, four dredger
models, and all eight FAQ topic categories.

Sources:
  (a) Staff interview questions (BDS-FAQ-001)
  (b) Specification lookups (BDS-SPEC-001)
  (c) Procedural and diagnostic queries (BDS-OM-001, BDS-TSG-001)
  (d) Pricing and commercial queries (BDS-PL-001)

Thesis reference: Section 8.1 (Evaluation Strategy).
"""

from __future__ import annotations
from dataclasses import dataclass
from collections import Counter


@dataclass
class EvalQuery:
    query_id      : str
    query         : str
    expected_model: str   # "Model 1"|"Model 2"|"Model 3"|"Model 4"|"All"
    topic_category: str
    relevant_doc  : str   # primary document_id for retrieval recall
    ground_truth  : str   # reference answer for RAGAS context recall


EVAL_QUERIES: list[EvalQuery] = [
    # ── Specification — Model 1 (16/14 CSD) ─────────────────────────────────
    EvalQuery("SPEC-M1-01","What is the engine fitted to the 16/14 inch cutter suction dredger?",
        "Model 1","equipment_knowledge","BDS-SPEC-001",
        "The 16/14 inch CSD is fitted with a Weichai X6170ZC650-2 main engine (478 kW / 650 HP) and a Caterpillar 3412 auxiliary engine."),
    EvalQuery("SPEC-M1-02","What is the maximum dredging depth of the 16/14 dredger?",
        "Model 1","equipment_knowledge","BDS-SPEC-001","Maximum dredging depth is 22 metres."),
    EvalQuery("SPEC-M1-03","What is the maximum discharge distance for the 16/14?",
        "Model 1","equipment_knowledge","BDS-SPEC-001","Maximum discharge distance is 1,000 metres."),
    EvalQuery("SPEC-M1-04","What is the normal pump operating pressure on the 16/14?",
        "Model 1","equipment_knowledge","BDS-SPEC-001","Normal operating pump pressure is 3 bar; maximum is 10 bar."),
    EvalQuery("SPEC-M1-05","What is the bulk production rate of the 16/14 dredger?",
        "Model 1","performance","BDS-SPEC-001","Maximum bulk production rate is 500 m³ per hour."),

    # ── Specification — Model 2 (14/12 Amphibious) ──────────────────────────
    EvalQuery("SPEC-M2-01","How deep can the 14/12 amphibious dredger dredge?",
        "Model 2","equipment_knowledge","BDS-SPEC-001","Maximum dredging depth is 22 metres."),
    EvalQuery("SPEC-M2-02","What is the discharge distance of the 14/12?",
        "Model 2","equipment_knowledge","BDS-SPEC-001","Maximum discharge distance is 600 metres."),
    EvalQuery("SPEC-M2-03","What engine does the 14/12 amphibious dredger use?",
        "Model 2","equipment_knowledge","BDS-SPEC-001",
        "The 14/12 uses a Weichai X6170ZC650-2 engine (478 kW / 650 HP) and a CAT 3412 auxiliary."),
    EvalQuery("SPEC-M2-04","What is the in-situ production rate of the 14/12?",
        "Model 2","performance","BDS-SPEC-001","Maximum in-situ production rate is 300 m³ per hour."),

    # ── Specification — Model 3 (12/10 Bucket Chain) ─────────────────────────
    EvalQuery("SPEC-M3-01","What engine is fitted to the 12/10 bucket chain dredger?",
        "Model 3","equipment_knowledge","BDS-SPEC-001",
        "The 12/10 Bucket Chain Dredger is fitted with a Caterpillar 3408 main engine."),
    EvalQuery("SPEC-M3-02","What is the bore and stroke of the CAT 3408?",
        "Model 3","equipment_knowledge","BDS-SPEC-001","Bore 137.2 mm, stroke 152.4 mm, displacement 18.02 litres."),
    EvalQuery("SPEC-M3-03","What is the in-situ production rate of the 12/10?",
        "Model 3","performance","BDS-SPEC-001","Maximum in-situ production rate is 200 m³ per hour."),
    EvalQuery("SPEC-M3-04","What oil capacity does the CAT 3408 require?",
        "Model 3","maintenance_practice","BDS-SPEC-001","Engine oil refill capacity is 46 litres."),

    # ── Specification — Model 4 (10/10 Jet Suction) ──────────────────────────
    EvalQuery("SPEC-M4-01","What drive system does the 10/10 jet suction dredger use?",
        "Model 4","equipment_knowledge","BDS-SPEC-001",
        "The 10/10 uses a PTO (Power Take-Off) drive, unlike the other models which use a marine gearbox."),
    EvalQuery("SPEC-M4-02","What is the maximum dredging depth of the 10/10?",
        "Model 4","equipment_knowledge","BDS-SPEC-001","Maximum dredging depth is 18 metres."),
    EvalQuery("SPEC-M4-03","What engine powers the 10/10 jet suction dredger?",
        "Model 4","equipment_knowledge","BDS-SPEC-001",
        "The 10/10 is powered by a Weichai WP13C550-18E121 engine (405 kW / 450 HP)."),
    EvalQuery("SPEC-M4-04","What is the discharge distance of the 10/10?",
        "Model 4","equipment_knowledge","BDS-SPEC-001","Maximum discharge distance is 300 metres."),

    # ── Cross-model ────────────────────────────────────────────────────────────
    EvalQuery("SPEC-CX-01","Which dredgers can reach 22 metres depth?",
        "All","equipment_knowledge","BDS-SPEC-001",
        "The 16/14 CSD and 14/12 Amphibious both reach 22 metres. The 10/10 is limited to 18 metres."),
    EvalQuery("SPEC-CX-02","What auxiliary engine do all four dredgers share?",
        "All","equipment_knowledge","BDS-SPEC-001",
        "All four models share the Caterpillar 3412 auxiliary engine (15 kVA, 240 V)."),
    EvalQuery("SPEC-CX-03","What onboard generator voltage is standard across all models?",
        "All","equipment_knowledge","BDS-SPEC-001","All models operate at 240 V with a 15 kVA generator."),
    EvalQuery("SPEC-CX-04","What is the difference between the 16/14 and 14/12 dredger?",
        "All","equipment_knowledge","BDS-SPEC-001",
        "The 16/14 has higher production (500 vs 300 m³/hr) and longer discharge (1000 vs 600 m). The 14/12 is amphibious."),

    # ── O&M — Startup & Shutdown ───────────────────────────────────────────────
    EvalQuery("OM-01","What is the engine startup sequence?",
        "All","daily_operations","BDS-OM-001",
        "Check oil, check gearbox, check fuel, check battery, set throttle to idle, turn on ignition. Allow 5 min warm-up before loading."),
    EvalQuery("OM-02","What must be done before starting every morning?",
        "All","daily_operations","BDS-OM-001",
        "Check oil, fuel, hose connections, safety guards, anchor lines, emergency stop, fire extinguisher, and PPE."),
    EvalQuery("OM-03","What is the shutdown procedure at end of shift?",
        "All","daily_operations","BDS-OM-001",
        "Flush pump with clean water for min 5 minutes, raise suction ladder, shut down engine after cool-down."),
    EvalQuery("OM-04","How do you flush the pump at end of shift?",
        "All","daily_operations","BDS-OM-001",
        "Open clean water supply to suction and run pump for minimum 5 minutes to clear all slurry."),
    EvalQuery("OM-05","How long should the engine warm up before loading?",
        "All","daily_operations","BDS-OM-001","Minimum 5 minutes; 10 minutes in cool conditions."),

    # ── O&M — Maintenance ─────────────────────────────────────────────────────
    EvalQuery("OM-06","How often should the engine oil be changed?",
        "All","maintenance_practice","BDS-OM-001","Every 200 operating hours."),
    EvalQuery("OM-07","What engine oil is recommended?",
        "All","maintenance_practice","BDS-OM-001","Total Ruby X engine oil."),
    EvalQuery("OM-08","How many grease points does the vessel have?",
        "All","maintenance_practice","BDS-OM-001","Seven grease points."),
    EvalQuery("OM-09","What maintenance is done at the 200-hour service?",
        "All","maintenance_practice","BDS-OM-001",
        "Oil and filter change, fuel filter replacement, gearbox oil, pump wear plates, cutter inspection, winch check, fastener torques, electrical connections."),
    EvalQuery("OM-10","What is the hull survey interval?",
        "All","maintenance_practice","BDS-OM-001","Annual (every 12 months)."),

    # ── O&M — Operating guidelines ─────────────────────────────────────────────
    EvalQuery("OM-11","What is the normal operating RPM during dredging?",
        "All","daily_operations","BDS-OM-001","1,200 to 2,100 RPM; maximum 1,800 RPM for marine-rated operation."),
    EvalQuery("OM-12","What is the maximum pump pressure?",
        "All","equipment_knowledge","BDS-OM-001","Maximum 10 bar; normal 3 bar."),
    EvalQuery("OM-13","What soil types are unsuitable for these dredgers?",
        "All","equipment_knowledge","BDS-OM-001","Hard rock and stiff clay. Coarse gravel above 150 mm can damage the impeller."),
    EvalQuery("OM-14","What is the engine temperature alarm threshold?",
        "All","daily_operations","BDS-OM-001","105 degrees Celsius — stop engine immediately."),

    # ── O&M — Safety & Emergency ───────────────────────────────────────────────
    EvalQuery("OM-15","What is the fire procedure on board?",
        "All","safety","BDS-OM-001",
        "Alert crew, activate emergency stop, use nearest extinguisher. If not controlled in 60 seconds, evacuate."),
    EvalQuery("OM-16","What to do if someone falls overboard?",
        "All","safety","BDS-OM-001","Shout man overboard, throw life ring from the bow, stop engine, deploy diver if needed."),
    EvalQuery("OM-17","Where is the life ring kept?",
        "All","safety","BDS-OM-001","At the head (bow) of the dredger."),

    # ── Troubleshooting ────────────────────────────────────────────────────────
    EvalQuery("TSG-01","The pump has lost suction — what should I check?",
        "All","fault_diagnosis","BDS-TSG-001",
        "Check suction head is submerged, check for air ingress at pipe joints, check impeller wear if over 200 hours."),
    EvalQuery("TSG-02","The engine produces black smoke — what is the cause?",
        "All","fault_diagnosis","BDS-TSG-001","Most likely a clogged air filter. Replace air filter first."),
    EvalQuery("TSG-03","The engine is overheating — what do I do?",
        "All","fault_diagnosis","BDS-TSG-001",
        "Stop engine immediately, allow 10 min cool-down, check fan, oil level, air intake, ventilation."),
    EvalQuery("TSG-04","The discharge pipe is blocked — how do I clear it?",
        "All","fault_diagnosis","BDS-TSG-001",
        "Stop pump immediately. Loosen joints from pump outlet to locate blockage. Clear manually or flush. Inspect flexible hoses."),
    EvalQuery("TSG-05","The pump seal is leaking — what should I do?",
        "All","fault_diagnosis","BDS-TSG-001",
        "2–3 drops/min is normal. A steady flow needs gland packing tightening or replacement (Level 2)."),
    EvalQuery("TSG-06","Low oil pressure warning — what should I do?",
        "All","fault_diagnosis","BDS-TSG-001","STOP ENGINE IMMEDIATELY. Check dipstick. Call specialist if level is correct."),
    EvalQuery("TSG-07","The engine keeps stalling when load is applied — why?",
        "All","fault_diagnosis","BDS-TSG-001","Fuel starvation (blocked filter or air in fuel), idle speed too low, or water in fuel."),
    EvalQuery("TSG-08","The gearbox is overheating — what is the cause?",
        "Model 3","fault_diagnosis","BDS-TSG-001","Low gearbox oil or discharge pipe back-pressure. Top up oil, check discharge, allow cool-down."),
    EvalQuery("TSG-09","Production has dropped significantly — how do I diagnose this?",
        "All","performance","BDS-TSG-001",
        "Check engine RPM, pump pressure, suction head position, and discharge pipe for partial blockage."),
    EvalQuery("TSG-10","The anchor wire snapped — what should I do?",
        "All","safety","BDS-TSG-001","Stop all dredging. Deploy remaining anchors. Replace wire before resuming."),
    EvalQuery("TSG-11","Engine knocking — should I keep running?",
        "All","fault_diagnosis","BDS-TSG-001","No. Stop immediately. Check oil. If knock returns on restart, call specialist."),
    EvalQuery("TSG-12","White smoke from exhaust — what does it indicate?",
        "All","fault_diagnosis","BDS-TSG-001","White smoke usually indicates water ingress into combustion — possible head gasket issue. Stop and inspect."),

    # ── FAQ — Staff interview knowledge ───────────────────────────────────────
    EvalQuery("FAQ-01","What should I check first every morning before starting?",
        "All","daily_operations","BDS-FAQ-001","Check oil first. If oil is low and you start, bearing damage is immediate. Then fuel, then gearbox."),
    EvalQuery("FAQ-02","Why does production drop in the afternoon?",
        "All","performance","BDS-FAQ-001",
        "Easy surface material is gone by afternoon, engine runs hotter, and operator fatigue means late repositioning."),
    EvalQuery("FAQ-03","How do I know when the impeller needs replacing?",
        "All","maintenance_practice","BDS-FAQ-001",
        "20% production drop at same RPM, impeller diameter 10% below spec, or rougher pump sound."),
    EvalQuery("FAQ-04","What is the most important maintenance task?",
        "All","maintenance_practice","BDS-FAQ-001","Engine oil change every 200 hours without exception."),
    EvalQuery("FAQ-05","What are the most common mistakes new operators make?",
        "All","daily_operations","BDS-FAQ-001",
        "Not flushing pump at end of shift, not checking anchor tension, running engine too hard."),
    EvalQuery("FAQ-06","Can these dredgers work in saltwater?",
        "All","equipment_knowledge","BDS-FAQ-001",
        "Yes, with high-chrome pump casing. Rinse with fresh water after every saltwater shift."),
    EvalQuery("FAQ-07","What realistic production should I expect from the 16/14?",
        "Model 1","performance","BDS-FAQ-001",
        "Plan at 70% of rated figure — 350 to 400 m³/hr consistently. Peak 500 m³/hr is achievable in ideal conditions."),
    EvalQuery("FAQ-08","How do I know when to reposition the anchors?",
        "All","daily_operations","BDS-FAQ-001","Watch discharge colour. When it looks lighter (more water), reposition."),
    EvalQuery("FAQ-09","What spare parts should always be kept on site?",
        "All","spare_parts","BDS-FAQ-001",
        "Engine oil (10L+), fuel filters, pump impeller, gland packing rope, anchor wire."),
    EvalQuery("FAQ-10","How do I tell if an engine knock is serious?",
        "All","fault_diagnosis","BDS-FAQ-001","All engine knock is serious. Stop immediately and investigate."),
    EvalQuery("FAQ-11","What fuel consumption can customers expect?",
        "All","customer_interactions","BDS-FAQ-001",
        "16/14: 40–55 L/hr. 10/10: 25–35 L/hr. Budget for upper end of range."),
    EvalQuery("FAQ-12","What should a first-time buyer know?",
        "All","customer_interactions","BDS-FAQ-001",
        "Buy training with the equipment. Set up 200-hour maintenance before first start. Keep spares on site from day one."),
    EvalQuery("FAQ-13","How many grease points daily and how much grease?",
        "All","maintenance_practice","BDS-FAQ-001","Seven points. Apply 2–3 shots per point; do not over-grease."),
    EvalQuery("FAQ-14","What is the 60-second fire rule?",
        "All","safety","BDS-FAQ-001","If fire is not out in 60 seconds, everyone leaves the vessel immediately."),
    EvalQuery("FAQ-15","Why is flushing the pump the most important shutdown task?",
        "All","daily_operations","BDS-FAQ-001",
        "Slurry hardens overnight if left in the pump and pipe. 5 minutes flushing prevents hours of downtime."),
    EvalQuery("FAQ-16","Is the CAT 3408 fuel system more sensitive than the Weichai?",
        "Model 3","maintenance_practice","BDS-FAQ-001",
        "Yes. The 3408 HEUI injectors are sensitive to water-contaminated diesel. Drain the water separator weekly."),
    EvalQuery("FAQ-17","Why is flushing critical for the 10/10?",
        "Model 4","daily_operations","BDS-FAQ-001",
        "The narrow 10-inch discharge pipe blocks more easily than larger models. Flushing is non-negotiable."),
    EvalQuery("FAQ-18","What is the oil change interval and why is it more frequent than OEM?",
        "All","maintenance_practice","BDS-FAQ-001",
        "200 hours. Benamdaj uses a more conservative interval than OEM because of harsh tropical river dredging conditions."),

    # ── Cross-document synthesis queries ──────────────────────────────────────
    EvalQuery("CX-01","Why does the 10/10 use PTO instead of a gearbox?",
        "Model 4","equipment_knowledge","BDS-FAQ-001",
        "PTO is simpler and lower cost for a smaller pump. Marine gearboxes are needed for heavier loads on larger models."),
    EvalQuery("CX-02","What happens if I don't flush the pump and how do I fix a blocked pipe?",
        "All","fault_diagnosis","BDS-TSG-001",
        "Slurry hardens overnight. Stop pump, loosen pipe joints from outlet, clear manually or flush with high-pressure water."),
    EvalQuery("CX-03","How do I diagnose low discharge density — mostly water?",
        "All","fault_diagnosis","BDS-TSG-001",
        "Lower the suction arm, reposition anchors to fresh material, confirm engine is at rated RPM."),
    EvalQuery("CX-04","What is the correct oil and interval for the engine?",
        "All","maintenance_practice","BDS-OM-001","Total Ruby X. Change every 200 hours."),
    EvalQuery("CX-05","How do I prepare the vessel for the annual hull survey?",
        "All","maintenance_practice","BDS-OM-001",
        "Schedule dry-dock or haul-out. Inspect underwater hull, anodes, shaft seals, and propulsion fittings."),

    # ── Price List ─────────────────────────────────────────────────────────────
    EvalQuery("PL-01","What is the purchase price of the 16/14 inch cutter suction dredger?",
        "Model 1","customer_interactions","BDS-PL-001",
        "Refer to the current price list for the BDS-CSD-1614 unit price."),
    EvalQuery("PL-02","How much does the 14/12 amphibious dredger cost?",
        "Model 2","customer_interactions","BDS-PL-001",
        "Refer to the current price list for the BDS-AMD-1412 unit price."),
    EvalQuery("PL-03","What is the price of the 12/10 bucket chain dredger?",
        "Model 3","customer_interactions","BDS-PL-001",
        "Refer to the current price list for the BDS-BCD-1210 unit price."),
    EvalQuery("PL-04","How much does the 10/10 jet suction dredger cost?",
        "Model 4","customer_interactions","BDS-PL-001",
        "Refer to the current price list for the BDS-JSD-1010 unit price."),
    EvalQuery("PL-05","What is the cost of a replacement pump impeller?",
        "All","spare_parts","BDS-PL-001",
        "Refer to the spare parts section of the price list for impeller pricing by model."),
    EvalQuery("PL-06","How much does a set of gland packing rope cost?",
        "All","spare_parts","BDS-PL-001",
        "Refer to the spare parts section of the price list for gland packing pricing."),
    EvalQuery("PL-07","What is the price of a replacement fuel filter?",
        "All","spare_parts","BDS-PL-001",
        "Refer to the spare parts section of the price list for fuel filter pricing."),
    EvalQuery("PL-08","How much does a 200-hour service package cost?",
        "All","maintenance_practice","BDS-PL-001",
        "Refer to the service packages section of the price list for the 200-hour service cost."),
    EvalQuery("PL-09","What are the payment terms for purchasing a dredger?",
        "All","customer_interactions","BDS-PL-001",
        "Refer to the commercial terms section of the price list for payment terms."),
    EvalQuery("PL-10","Is there a discount for purchasing multiple units?",
        "All","customer_interactions","BDS-PL-001",
        "Refer to the commercial terms section of the price list for volume discount policy."),
    EvalQuery("PL-11","What is the delivery cost for a dredger within Nigeria?",
        "All","customer_interactions","BDS-PL-001",
        "Refer to the logistics section of the price list for domestic delivery charges."),
    EvalQuery("PL-12","Does Benamdaj offer equipment hire or rental?",
        "All","customer_interactions","BDS-PL-001",
        "Refer to the price list for hire/rental terms and daily or monthly rates."),
    EvalQuery("PL-13","What is the cost of operator training for a new dredger purchase?",
        "All","customer_interactions","BDS-PL-001",
        "Refer to the training and support section of the price list for training package costs."),
    EvalQuery("PL-14","How much does anchor wire replacement cost?",
        "All","spare_parts","BDS-PL-001",
        "Refer to the spare parts section of the price list for anchor wire pricing."),
    EvalQuery("PL-15","What is the price of a replacement pump wear plate?",
        "All","spare_parts","BDS-PL-001",
        "Refer to the spare parts section of the price list for wear plate pricing by model."),
    EvalQuery("PL-16","Is warranty included in the purchase price?",
        "All","customer_interactions","BDS-PL-001",
        "Refer to the warranty and after-sales section of the price list for warranty terms."),
    EvalQuery("PL-17","What does the annual maintenance contract include and how much does it cost?",
        "All","maintenance_practice","BDS-PL-001",
        "Refer to the service packages section of the price list for annual contract scope and pricing."),
    EvalQuery("PL-18","How much does a CAT 3408 engine oil filter cost?",
        "Model 3","spare_parts","BDS-PL-001",
        "Refer to the spare parts section of the price list for CAT 3408 filter pricing."),
    EvalQuery("PL-19","What is the export price for the 16/14 dredger?",
        "Model 1","customer_interactions","BDS-PL-001",
        "Refer to the export pricing section of the price list for CIF or FOB price of the BDS-CSD-1614."),
    EvalQuery("PL-20","Are spare parts prices fixed or subject to negotiation?",
        "All","customer_interactions","BDS-PL-001",
        "Refer to the commercial terms section of the price list for spare parts pricing policy."),
]


def get_queries_by_document(doc_id: str) -> list[EvalQuery]:
    return [q for q in EVAL_QUERIES if q.relevant_doc == doc_id]

def get_queries_by_model(model: str) -> list[EvalQuery]:
    return [q for q in EVAL_QUERIES if q.expected_model in (model, "All")]

def get_queries_by_topic(topic: str) -> list[EvalQuery]:
    return [q for q in EVAL_QUERIES if q.topic_category == topic]

def balanced_sample(per_doc: int = 4) -> list[EvalQuery]:
    """
    Return a balanced subset with exactly `per_doc` queries from each
    document group. Queries are taken in list order (deterministic).
    With 5 documents and per_doc=4 the default sample is 20 queries.
    """
    groups: dict[str, list[EvalQuery]] = {}
    for q in EVAL_QUERIES:
        groups.setdefault(q.relevant_doc, []).append(q)
    sample: list[EvalQuery] = []
    for doc_id in sorted(groups):
        sample.extend(groups[doc_id][:per_doc])
    return sample

def summary() -> dict:
    return {
        "total"       : len(EVAL_QUERIES),
        "by_document" : dict(Counter(q.relevant_doc for q in EVAL_QUERIES)),
        "by_model"    : dict(Counter(q.expected_model for q in EVAL_QUERIES)),
        "by_topic"    : dict(Counter(q.topic_category for q in EVAL_QUERIES)),
    }
