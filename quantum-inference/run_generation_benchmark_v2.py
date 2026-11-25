"""Run the expanded v2 held-out benchmark against Quantum Candidate only."""
import argparse
import json
import os
from pathlib import Path

from run_generation_benchmark import run_benchmark


DEFAULT_BENCHMARK_V2 = Path(__file__).resolve().parent / "evaluation" / "benchmark_v2.json"


def main():
    parser = argparse.ArgumentParser(
        description="Run the 91-case held-out Korean benchmark against Quantum Candidate."
    )
    parser.add_argument("--url", default=os.getenv("QUANTUM_CANDIDATE_URL", ""))
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK_V2))
    parser.add_argument("--output", default="quantum-candidate-generation-evaluation-v2.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    report = run_benchmark(
        args.url,
        os.getenv("QUANTUM_API_TOKEN", ""),
        benchmark_path=args.benchmark,
        output_path=args.output,
        limit=args.limit,
    )
    print(json.dumps({
        "benchmark_id": report["benchmark_id"],
        "benchmark_version": report["benchmark_version"],
        "cases_requested": report["cases_requested"],
        "cases_completed": report["cases_completed"],
        "generation_invocations": report["generation_invocations"],
        "approved_qa_retrieval_cases": report["approved_qa_retrieval_cases"],
        "human_review_required": report["human_review_required"],
        "automatic_quality_score": None,
        "model_training": False,
        "production_model_touched": False,
        "output_path": report["output_path"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
