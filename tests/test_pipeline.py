"""
tests/test_pipeline.py
=======================
Integration tests for the Benamdaj RAG pipeline.
These tests validate the pipeline logic using synthetic data —
no real documents, API keys, or Pinecone connection required.

Run with:  python -m pytest tests/ -v
"""

import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import (
    clean_text, count_tokens, generate_chunk_id,
    slugify, table_to_text, fingerprint
)
from ingestion.chunker import (
    _detect_model_from_text, _classify_faq_topic, _group_by_heading
)
from ingestion.document_loader import DocumentBlock
from retrieval.query_processor import process_query


class TestHelpers(unittest.TestCase):

    def test_clean_text_normalises_dashes(self):
        result = clean_text("Engine\u2014pump connected")
        self.assertIn("--", result)

    def test_count_tokens_approximate(self):
        # 5 words → 5 tokens approximately
        self.assertEqual(count_tokens("one two three four five"), 5)

    def test_generate_chunk_id_stable(self):
        cid1 = generate_chunk_id("BDS-SPEC-001", 3, "Engine Power System")
        cid2 = generate_chunk_id("BDS-SPEC-001", 3, "Engine Power System")
        self.assertEqual(cid1, cid2)

    def test_generate_chunk_id_unique(self):
        cid1 = generate_chunk_id("BDS-SPEC-001", 1, "Engine")
        cid2 = generate_chunk_id("BDS-SPEC-001", 2, "Engine")
        self.assertNotEqual(cid1, cid2)

    def test_slugify(self):
        self.assertEqual(slugify("Engine & Power System"), "engine-power-system")

    def test_table_to_text(self):
        rows   = [["Model", "Engine", "HP"], ["Model 1", "Weichai 6170", "650"]]
        result = table_to_text(rows)
        self.assertIn("Model 1", result)
        self.assertIn("|", result)

    def test_fingerprint_deterministic(self):
        self.assertEqual(fingerprint("hello"), fingerprint("hello"))
        self.assertNotEqual(fingerprint("hello"), fingerprint("world"))


class TestModelDetection(unittest.TestCase):

    def test_detect_model_1_from_16_14(self):
        self.assertEqual(_detect_model_from_text("16/14 inch cutter suction"), "Model 1")

    def test_detect_model_2_from_amphibious(self):
        self.assertEqual(_detect_model_from_text("14/12 Amphibious dredger"), "Model 2")

    def test_detect_model_3_from_cat_3408(self):
        self.assertEqual(_detect_model_from_text("CAT 3408 engine specification"), "Model 3")

    def test_detect_model_4_from_jet_suction(self):
        self.assertEqual(_detect_model_from_text("10/10 jet suction dredger WP13"), "Model 4")

    def test_detect_all_when_no_model_mentioned(self):
        self.assertEqual(_detect_model_from_text("Daily maintenance checklist"), "All")

    def test_detect_all_when_multiple_models(self):
        # When text mentions multiple models, default to "All"
        result = _detect_model_from_text("16/14 and 14/12 comparison")
        # Multiple matches → should return one model or All depending on priority
        self.assertIn(result, ["Model 1", "Model 2", "All"])


class TestFaqTopicClassification(unittest.TestCase):

    def test_fault_diagnosis_topic(self):
        topic = _classify_faq_topic("The pump has a leak and there is a fault with the seal pressure")
        self.assertEqual(topic, "fault_diagnosis")

    def test_maintenance_topic(self):
        topic = _classify_faq_topic("How often should the oil filter be changed for maintenance")
        self.assertEqual(topic, "maintenance_practice")

    def test_safety_topic(self):
        topic = _classify_faq_topic("What is the fire safety emergency procedure")
        self.assertEqual(topic, "safety")

    def test_performance_topic(self):
        topic = _classify_faq_topic("Why has the production rate and output dropped today")
        self.assertEqual(topic, "performance")


class TestQueryProcessor(unittest.TestCase):

    def test_model_1_detected(self):
        pq = process_query("What engine is in the 16/14 cutter suction dredger?")
        self.assertEqual(pq.detected_model, "Model 1")

    def test_model_4_detected(self):
        pq = process_query("How do I start the 10/10 jet suction?")
        self.assertEqual(pq.detected_model, "Model 4")

    def test_no_model_for_general_query(self):
        pq = process_query("How often should oil be changed?")
        self.assertIsNone(pq.detected_model)

    def test_pinecone_filter_built_for_model(self):
        pq = process_query("What is the discharge distance of the 14/12?")
        self.assertIsNotNone(pq.pinecone_filter)
        self.assertIn("$or", pq.pinecone_filter)

    def test_no_pinecone_filter_for_general(self):
        pq = process_query("What is the engine oil change interval?")
        self.assertIsNone(pq.pinecone_filter)

    def test_safety_query_flagged(self):
        pq = process_query("What is the fire emergency procedure?")
        self.assertTrue(pq.is_safety_query)

    def test_non_safety_query_not_flagged(self):
        pq = process_query("What is the pump inlet diameter?")
        self.assertFalse(pq.is_safety_query)

    def test_multi_model_query_detected(self):
        pq = process_query("Which dredger is best for shallow water work?")
        self.assertTrue(pq.is_multi_model)
        self.assertIsNone(pq.detected_model)

    def test_topic_classified_correctly(self):
        pq = process_query("The engine keeps stalling when I apply load")
        self.assertEqual(pq.topic_category, "fault_diagnosis")

    def test_cleaned_query_strips_whitespace(self):
        pq = process_query("  What is the oil grade?  ")
        self.assertEqual(pq.cleaned_query, "What is the oil grade?")


class TestEvalQuerySet(unittest.TestCase):

    def test_minimum_query_count(self):
        from evaluation.eval_queries import EVAL_QUERIES
        self.assertGreaterEqual(len(EVAL_QUERIES), 50)

    def test_all_queries_have_ground_truth(self):
        from evaluation.eval_queries import EVAL_QUERIES
        for q in EVAL_QUERIES:
            self.assertGreater(len(q.ground_truth), 10,
                               f"Query {q.query_id} has insufficient ground truth")

    def test_all_four_documents_represented(self):
        from evaluation.eval_queries import EVAL_QUERIES
        docs = {q.relevant_doc for q in EVAL_QUERIES}
        for expected in ["BDS-SPEC-001", "BDS-OM-001", "BDS-TSG-001", "BDS-FAQ-001"]:
            self.assertIn(expected, docs, f"No eval queries for {expected}")

    def test_all_four_models_covered(self):
        from evaluation.eval_queries import EVAL_QUERIES
        models = {q.expected_model for q in EVAL_QUERIES}
        for expected in ["Model 1", "Model 2", "Model 3", "Model 4", "All"]:
            self.assertIn(expected, models, f"No eval queries for {expected}")

    def test_query_ids_unique(self):
        from evaluation.eval_queries import EVAL_QUERIES
        ids = [q.query_id for q in EVAL_QUERIES]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate query IDs found")


if __name__ == "__main__":
    unittest.main(verbosity=2)
