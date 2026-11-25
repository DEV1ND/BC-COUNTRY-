"""Add reproducible provenance and category diagnostics to a held-out run.

This post-processor never contacts a model, trains weights, or grades meaning.
It creates a new report and refuses to overwrite an existing destination.
"""
import argparse
import hashlib
import json
from pathlib import Path


GENERATION_CONFIG = {
    "max_new_tokens": 128,
    "temperature": 0.0,
    "top_k": 20,
    "repetition_penalty": 1.08,
}


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def enrich_generation_report(report, benchmark_bytes, generation_config=None):
    if not isinstance(report, dict):
        raise ValueError("Generation report must be a JSON object.")
    if report.get("training_eligible") is not False:
        raise ValueError("Generation report must be training_eligible=false.")
    if report.get("production_model_touched") is not False:
        raise ValueError("Generation report must explicitly preserve Production.")
    if report.get("model_training") is not False:
        raise ValueError("Generation report must explicitly state model_training=false.")
    if not isinstance(benchmark_bytes, bytes):
        raise ValueError("benchmark_bytes must be raw bytes.")
    try:
        dataset = json.loads(benchmark_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Benchmark must be valid UTF-8 JSON.") from exc
    if not isinstance(dataset, dict):
        raise ValueError("Benchmark must be a JSON object.")
    if dataset.get("purpose") != "held_out_evaluation_only":
        raise ValueError("Only held-out evaluation datasets are accepted.")
    if dataset.get("training_eligible") is not False:
        raise ValueError("Benchmark must be training_eligible=false.")
    if dataset.get("benchmark_id") != report.get("benchmark_id"):
        raise ValueError("Benchmark ID does not match the generation report.")
    if dataset.get("version") != report.get("benchmark_version"):
        raise ValueError("Benchmark version does not match the generation report.")

    cases = dataset.get("cases")
    results = report.get("results")
    if not isinstance(cases, list) or not isinstance(results, list) or not results:
        raise ValueError("Benchmark cases and report results must be non-empty arrays.")
    expected = {}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise ValueError("Every benchmark case must have a string ID.")
        if case["id"] in expected:
            raise ValueError("Benchmark contains duplicate case IDs.")
        expected[case["id"]] = case

    seen = set()
    category_stats = {}
    enriched_results = []
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("Every generation result must be an object.")
        case_id = result.get("id")
        if not isinstance(case_id, str) or case_id not in expected or case_id in seen:
            raise ValueError("Report contains an unknown or duplicate case ID.")
        source_case = expected[case_id]
        if result.get("category") != source_case.get("category"):
            raise ValueError("Report category does not match the benchmark.")
        if result.get("prompt") != source_case.get("prompt"):
            raise ValueError("Report prompt does not match the benchmark.")
        answer = result.get("answer")
        if not isinstance(answer, str):
            raise ValueError("Every result must contain an answer string.")
        diagnostics = result.get("output_diagnostics")
        if not isinstance(diagnostics, dict):
            raise ValueError("Every result must contain output diagnostics.")
        seen.add(case_id)

        prompt_material = (
            (result.get("context") + "\\n\\n") if result.get("context") else ""
        ) + result["prompt"]
        item = dict(result)
        item["provenance"] = {
            "prompt_sha256": _sha256(prompt_material.encode("utf-8")),
            "answer_sha256": _sha256(answer.encode("utf-8")),
            "answer_characters": len(answer),
            "answer_utf8_bytes": len(answer.encode("utf-8")),
        }
        enriched_results.append(item)

        category = result["category"]
        bucket = category_stats.setdefault(category, {
            "cases": 0,
            "model_generation_cases": 0,
            "approved_qa_retrieval_cases": 0,
            "empty_answers": 0,
            "fragmentation_flags": 0,
            "repetition_flags": 0,
            "answer_characters_total": 0,
            "answer_characters_min": None,
            "answer_characters_max": 0,
        })
        answer_length = len(answer)
        bucket["cases"] += 1
        bucket["model_generation_cases"] += int(
            result.get("response_mode") == "model_generation"
        )
        bucket["approved_qa_retrieval_cases"] += int(
            result.get("response_mode") == "approved_qa_retrieval"
        )
        bucket["empty_answers"] += int(not answer.strip())
        bucket["fragmentation_flags"] += int(
            diagnostics.get("suspicious_fragmentation") is True
        )
        bucket["repetition_flags"] += int(
            diagnostics.get("suspicious_repetition") is True
        )
        bucket["answer_characters_total"] += answer_length
        current_min = bucket["answer_characters_min"]
        bucket["answer_characters_min"] = (
            answer_length if current_min is None else min(current_min, answer_length)
        )
        bucket["answer_characters_max"] = max(
            bucket["answer_characters_max"], answer_length
        )

    for bucket in category_stats.values():
        bucket["answer_characters_average"] = round(
            bucket["answer_characters_total"] / bucket["cases"], 2
        )

    ids_in_order = [result["id"] for result in results]
    config = GENERATION_CONFIG if generation_config is None else generation_config
    if not isinstance(config, dict):
        raise ValueError("generation_config must be an object.")

    enriched = dict(report)
    enriched["report_schema"] = "juckex-quantum-evaluation-provenance-v2"
    enriched["evaluation_provenance"] = {
        "benchmark_sha256": _sha256(benchmark_bytes),
        "benchmark_case_count": len(cases),
        "evaluated_case_count": len(results),
        "evaluated_case_ids_sha256": _sha256(_canonical_json(ids_in_order)),
        "generation_config": dict(config),
        "category_diagnostics": dict(sorted(category_stats.items())),
        "interpretation": (
            "Hashes establish artifact identity; category metrics describe visible "
            "output shape only and are not semantic quality scores."
        ),
        "automatic_quality_score": None,
        "human_review_required": True,
        "training_eligible": False,
        "production_model_touched": False,
    }
    enriched["results"] = enriched_results
    return enriched


def main():
    parser = argparse.ArgumentParser(
        description="Enrich a Quantum Candidate held-out report with reproducible provenance."
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report_path = Path(args.report)
    benchmark_path = Path(args.benchmark)
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite an existing enriched report.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    enriched = enrich_generation_report(
        report, benchmark_path.read_bytes(), GENERATION_CONFIG
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(enriched, ensure_ascii=False, indent=2) + "\\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "evaluation_report_enriched",
        "report_schema": enriched["report_schema"],
        "benchmark_sha256": enriched["evaluation_provenance"]["benchmark_sha256"],
        "evaluated_case_count": enriched["evaluation_provenance"]["evaluated_case_count"],
        "category_count": len(enriched["evaluation_provenance"]["category_diagnostics"]),
        "automatic_quality_score": None,
        "training_eligible": False,
        "production_model_touched": False,
        "output": str(destination),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
