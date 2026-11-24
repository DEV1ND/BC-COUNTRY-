"""Compare current lexical ranking with an isolated n-gram experiment.

Requires a locally exported, provenance-reviewed corpus JSON array. It does not
download data, call an LLM, modify the index, train weights, or promote a model.
"""
import argparse
import json
from pathlib import Path

from experimental_retrieval import rank_experimental
from knowledge_retrieval import rank_documents
from retrieval_benchmark import evaluate_retrieval_cases


def load_json(path, label):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} 파일을 읽을 수 없습니다.") from exc
    return value


def compare_corpus(corpus, cases, k=5):
    if not isinstance(corpus, list):
        raise ValueError("corpus는 JSON 객체 배열이어야 합니다.")
    if not isinstance(cases, list):
        raise ValueError("cases는 JSON 배열이어야 합니다.")
    # Validate corpus once before invoking either algorithm.
    for index, row in enumerate(corpus):
        if not isinstance(row, dict):
            raise ValueError(f"corpus 문서 {index}는 객체여야 합니다.")
        if not all(isinstance(row.get(key), str) for key in ("title", "text")):
            raise ValueError(f"corpus 문서 {index}의 title/text가 올바르지 않습니다.")
        if not isinstance(row.get("url", ""), str):
            raise ValueError(f"corpus 문서 {index}의 url이 올바르지 않습니다.")

    baseline = evaluate_retrieval_cases(
        cases,
        lambda question, limit: rank_documents(question, corpus, limit=limit),
        k=k,
    )
    experiment = evaluate_retrieval_cases(
        cases,
        lambda question, limit: rank_experimental(question, corpus, limit=limit),
        k=k,
    )
    return {
        "scope": "offline_retrieval_comparison_only",
        "corpus_documents": len(corpus),
        "evaluation_cases": len(cases),
        "baseline": baseline,
        "experimental_character_ngram_tfidf": experiment,
        "delta": {
            "hit_rate_at_k": round(experiment["hit_rate_at_k"] - baseline["hit_rate_at_k"], 4),
            "precision_at_returned": round(
                experiment["precision_at_returned"] - baseline["precision_at_returned"], 4
            ),
            "mean_reciprocal_rank": round(
                experiment["mean_reciprocal_rank"] - baseline["mean_reciprocal_rank"], 4
            ),
            "retrieval_noise_rate": (
                round(experiment["retrieval_noise_rate"] - baseline["retrieval_noise_rate"], 4)
                if experiment["retrieval_noise_rate"] is not None
                and baseline["retrieval_noise_rate"] is not None else None
            ),
            "false_positive_retrievals": (
                experiment["false_positive_retrievals"] - baseline["false_positive_retrievals"]
            ),
        },
        "promotion_eligible": False,
        "semantic_correctness_claimed": False,
        "model_invoked": False,
        "model_training": False,
        "production_model_touched": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="Reviewed local corpus JSON array")
    parser.add_argument(
        "--cases",
        default=str(Path(__file__).resolve().parent / "evaluation" / "knowledge_retrieval_cases.json"),
    )
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    corpus = load_json(args.corpus, "corpus")
    dataset = load_json(args.cases, "evaluation")
    cases = dataset.get("cases") if isinstance(dataset, dict) else None
    report = compare_corpus(corpus, cases, k=args.k)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
