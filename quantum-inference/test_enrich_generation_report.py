"""Tests for deterministic held-out evaluation report enrichment."""
import hashlib
import json
import unittest

from enrich_generation_report import enrich_generation_report


def fixture():
    dataset = {
        "benchmark_id": "quantum-eval-test",
        "version": 1,
        "purpose": "held_out_evaluation_only",
        "training_eligible": False,
        "categories": ["everyday_dialogue", "math"],
        "cases": [
            {"id": "ED-01", "category": "everyday_dialogue", "prompt": "안녕!", "context": ""},
            {"id": "MA-01", "category": "math", "prompt": "2 더하기 3은?", "context": "기초 산술"},
        ],
    }
    report = {
        "benchmark_id": "quantum-eval-test",
        "benchmark_version": 1,
        "training_eligible": False,
        "model_training": False,
        "production_model_touched": False,
        "results": [
            {
                "id": "ED-01",
                "category": "everyday_dialogue",
                "prompt": "안녕!",
                "context": None,
                "answer": "안녕하세요!",
                "response_mode": "model_generation",
                "output_diagnostics": {
                    "suspicious_fragmentation": False,
                    "suspicious_repetition": False,
                },
            },
            {
                "id": "MA-01",
                "category": "math",
                "prompt": "2 더하기 3은?",
                "context": "기초 산술",
                "answer": "5입니다.",
                "response_mode": "approved_qa_retrieval",
                "output_diagnostics": {
                    "suspicious_fragmentation": False,
                    "suspicious_repetition": False,
                },
            },
        ],
    }
    return dataset, report, json.dumps(dataset, ensure_ascii=False).encode("utf-8")


class EvaluationProvenanceTests(unittest.TestCase):
    def test_adds_dataset_fingerprint_and_case_hashes(self):
        dataset, report, raw = fixture()
        enriched = enrich_generation_report(report, raw)
        provenance = enriched["evaluation_provenance"]
        self.assertEqual(provenance["benchmark_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(provenance["evaluated_case_count"], 2)
        self.assertEqual(
            enriched["results"][0]["provenance"]["answer_sha256"],
            hashlib.sha256("안녕하세요!".encode("utf-8")).hexdigest(),
        )
        self.assertEqual(enriched["report_schema"], "juckex-quantum-evaluation-provenance-v2")

    def test_category_metrics_separate_model_generation_and_retrieval(self):
        _, report, raw = fixture()
        categories = enrich_generation_report(report, raw)["evaluation_provenance"]["category_diagnostics"]
        self.assertEqual(categories["everyday_dialogue"]["model_generation_cases"], 1)
        self.assertEqual(categories["math"]["approved_qa_retrieval_cases"], 1)
        self.assertEqual(categories["math"]["answer_characters_min"], len("5입니다."))

    def test_rejects_training_eligible_report(self):
        _, report, raw = fixture()
        report["training_eligible"] = True
        with self.assertRaisesRegex(ValueError, "training_eligible"):
            enrich_generation_report(report, raw)

    def test_rejects_production_touched_report(self):
        _, report, raw = fixture()
        report["production_model_touched"] = True
        with self.assertRaisesRegex(ValueError, "preserve Production"):
            enrich_generation_report(report, raw)

    def test_rejects_benchmark_identity_mismatch(self):
        dataset, report, _ = fixture()
        dataset["version"] = 2
        raw = json.dumps(dataset).encode("utf-8")
        with self.assertRaisesRegex(ValueError, "version"):
            enrich_generation_report(report, raw)

    def test_rejects_result_prompt_tampering(self):
        _, report, raw = fixture()
        report["results"][0]["prompt"] = "다른 질문"
        with self.assertRaisesRegex(ValueError, "prompt"):
            enrich_generation_report(report, raw)

    def test_rejects_duplicate_result_ids(self):
        _, report, raw = fixture()
        report["results"].append(dict(report["results"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            enrich_generation_report(report, raw)


if __name__ == "__main__":
    unittest.main()
