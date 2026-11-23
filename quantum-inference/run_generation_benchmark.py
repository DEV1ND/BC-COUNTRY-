"""Run the held-out answer benchmark against Quantum Candidate only.

This tool never trains, updates weights, or contacts an external language model.
Answers are captured for human review; rubric items are not auto-scored.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request


DEFAULT_BENCHMARK = Path(__file__).resolve().parent / "evaluation" / "benchmark_v1.json"


def _request_json(url, method, api_token, payload=None, timeout=90):
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Accept": "application/json", "User-Agent": "JUCKEX-Quantum-Evaluator"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if api_token:
        headers["Authorization"] = "Bearer " + api_token
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(2 * 1024 * 1024 + 1)
    except urllib.error.HTTPError as exc:
        raise RuntimeError("Candidate API returned HTTP " + str(exc.code) + ".") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError("Candidate API request failed.") from exc
    if len(raw) > 2 * 1024 * 1024:
        raise RuntimeError("Candidate API response exceeded the 2 MiB safety limit.")
    try:
        result = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Candidate API returned invalid JSON.") from exc
    if not isinstance(result, dict):
        raise RuntimeError("Candidate API response must be a JSON object.")
    return result


def validate_benchmark_dataset(dataset):
    """Validate the complete held-out dataset before contacting Candidate."""
    if not isinstance(dataset, dict) or dataset.get("purpose") != "held_out_evaluation_only":
        raise ValueError("Only the held-out evaluation benchmark may be used.")
    if dataset.get("training_eligible") is not False:
        raise ValueError("Benchmark must explicitly be marked training_eligible=false.")
    benchmark_id = dataset.get("benchmark_id")
    version = dataset.get("version")
    categories = dataset.get("categories")
    cases = dataset.get("cases")
    if not isinstance(benchmark_id, str) or not benchmark_id.strip():
        raise ValueError("Benchmark must have a non-empty benchmark_id.")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("Benchmark version must be a positive integer.")
    if not isinstance(categories, list) or not categories or any(
        not isinstance(item, str) or not item.strip() for item in categories
    ):
        raise ValueError("Benchmark categories must be a non-empty string list.")
    if len(categories) != len(set(categories)):
        raise ValueError("Benchmark categories must be unique.")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Benchmark must contain at least one evaluation case.")
    ids = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError("Benchmark case " + str(index) + " must be an object.")
        case_id = case.get("id")
        category = case.get("category")
        prompt = case.get("prompt")
        context = case.get("context", "")
        rubric = case.get("must_include")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("Every benchmark case must have a non-empty string ID.")
        if not isinstance(category, str) or category not in categories:
            raise ValueError("Benchmark case " + case_id + " has an unknown category.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Benchmark case " + case_id + " has an empty or invalid prompt.")
        if not isinstance(context, str):
            raise ValueError("Benchmark case " + case_id + " context must be a string.")
        if not isinstance(rubric, list) or any(
            not isinstance(item, str) or not item.strip() for item in rubric
        ):
            raise ValueError("Benchmark case " + case_id + " rubric must be a string list.")
        ids.append(case_id.strip())
    if len(ids) != len(set(ids)):
        raise ValueError("Benchmark cases must have unique non-empty IDs.")
    return cases


def run_benchmark(base_url, api_token, benchmark_path=DEFAULT_BENCHMARK, output_path=None, limit=None):
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("Candidate URL must be a non-empty string.")
    if not isinstance(api_token, str) or not api_token:
        raise ValueError("QUANTUM_API_TOKEN is required; do not put it in source files.")
    parsed = urllib.parse.urlsplit(base_url.strip())
    if parsed.scheme not in ("https", "http") or not parsed.netloc:
        raise ValueError("Candidate URL must be an absolute HTTP(S) URL.")
    if parsed.scheme != "https" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Non-local Candidate URLs must use HTTPS.")
    base = base_url.rstrip("/")

    try:
        dataset = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Could not read the held-out benchmark JSON.") from exc
    cases = validate_benchmark_dataset(dataset)
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer.")
        cases = cases[:limit]

    health = _request_json(base + "/health", "GET", api_token)
    if health.get("generation_ready") is not True or health.get("model_loaded") is not True:
        raise RuntimeError("Quantum Candidate is not generation-ready; no benchmark prompts were sent.")

    started = datetime.now(timezone.utc).isoformat()
    results = []
    generation_invocations = 0
    approved_qa_retrieval_cases = 0
    for case in cases:
        prompt = case["prompt"].strip()
        context = case.get("context", "").strip()
        if not prompt:
            raise ValueError("Benchmark case " + str(case["id"]) + " has an empty prompt.")
        full_prompt = (context + "\n\n" if context else "") + prompt
        response = _request_json(
            base + "/v1/generate",
            "POST",
            api_token,
            {"prompt": full_prompt, "temperature": 0.0, "max_new_tokens": 128},
        )
        answer = response.get("answer")
        if not isinstance(answer, str):
            raise RuntimeError("Candidate response for " + str(case["id"]) + " has no answer string.")
        if response.get("is_live_inference") is not True:
            raise RuntimeError("Candidate response for " + str(case["id"]) + " is not marked as live inference.")
        if response.get("model") not in ("quantum-approved-knowledge", "quantum-bpe-candidate"):
            raise RuntimeError("Candidate returned an unrecognized model identity for " + str(case["id"]) + ".")
        response_mode = response.get("response_mode")
        if response_mode == "approved_qa_retrieval":
            if response.get("model") != "quantum-approved-knowledge" or response.get("is_model_generation") is not False:
                raise RuntimeError("Candidate approved-QA response metadata is inconsistent for " + str(case["id"]) + ".")
            approved_qa_retrieval_cases += 1
        elif response_mode == "model_generation":
            if response.get("model") != "quantum-bpe-candidate" or response.get("is_model_generation") is not True:
                raise RuntimeError("Candidate model-generation metadata is inconsistent for " + str(case["id"]) + ".")
            generation_invocations += 1
        else:
            raise RuntimeError("Candidate returned an unknown response mode for " + str(case["id"]) + ".")
        results.append({
            "id": case["id"],
            "category": case["category"],
            "prompt": prompt,
            "context": context or None,
            "answer": answer,
            "response_model": response.get("model"),
            "response_mode": response_mode,
            "must_include_rubric": case.get("must_include", []),
            "human_review_status": "pending",
        })

    report = {
        "benchmark_id": dataset.get("benchmark_id"),
        "benchmark_version": dataset.get("version"),
        "purpose": dataset.get("purpose"),
        "training_eligible": False,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "candidate_url": base,
        "model_metadata": {
            "model": health.get("model"),
            "model_parameters": health.get("model_parameters"),
            "model_architecture": health.get("model_architecture"),
            "checkpoint_source": health.get("checkpoint_source"),
            "checkpoint_release_tag": health.get("checkpoint_release_tag"),
        },
        "cases_requested": len(cases),
        "cases_completed": len(results),
        "human_review_required": True,
        "automatic_quality_score": None,
        "results": results,
        "model_invoked": generation_invocations > 0,
        "generation_invocations": generation_invocations,
        "approved_qa_retrieval_cases": approved_qa_retrieval_cases,
        "model_training": False,
        "production_model_touched": False,
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["output_path"] = str(destination)
    return report


def main():
    parser = argparse.ArgumentParser(description="Run the held-out benchmark against Quantum Candidate.")
    parser.add_argument("--url", default=os.getenv("QUANTUM_CANDIDATE_URL", ""))
    parser.add_argument("--output", default="quantum-candidate-evaluation.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    report = run_benchmark(
        args.url,
        os.getenv("QUANTUM_API_TOKEN", ""),
        output_path=args.output,
        limit=args.limit,
    )
    print(json.dumps({
        "cases_completed": report["cases_completed"],
        "generation_invocations": report["generation_invocations"],
        "approved_qa_retrieval_cases": report["approved_qa_retrieval_cases"],
        "human_review_required": report["human_review_required"],
        "automatic_quality_score": report["automatic_quality_score"],
        "output_path": report["output_path"],
        "model_training": report["model_training"],
        "production_model_touched": report["production_model_touched"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
