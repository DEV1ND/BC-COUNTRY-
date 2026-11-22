"""Deterministic retrieval benchmark metrics; does not score or train model answers."""
from typing import Callable, Iterable, Mapping


def evaluate_retrieval_cases(
    cases: Iterable[Mapping],
    retrieve: Callable[[str, int], list],
    k: int = 5,
) -> dict:
    """Measure expected-title retrieval and safe abstention on a fixed case set."""
    k = max(1, min(int(k), 10))
    rows = list(cases)
    if not rows:
        raise ValueError("At least one evaluation case is required.")

    results = []
    hits = 0
    reciprocal_ranks = []
    precision_sum = 0.0
    abstention_correct = 0

    for index, case in enumerate(rows):
        if not isinstance(case, Mapping):
            raise ValueError(f"Case {index} must be an object.")
        question = str(case.get("question") or "").strip()
        domain = str(case.get("domain") or "").strip()
        expected = case.get("expected_titles")
        if not question or not domain or not isinstance(expected, list):
            raise ValueError(f"Case {index} requires domain, question, and expected_titles.")
        expected_titles = {str(title).strip().casefold() for title in expected if str(title).strip()}
        retrieved = retrieve(question, k)
        if not isinstance(retrieved, list):
            raise ValueError(f"Retriever returned a non-list for case {index}.")
        retrieved = retrieved[:k]
        titles = [str(item.get("title") or "").strip() for item in retrieved if isinstance(item, Mapping)]
        matched_positions = [
            position for position, title in enumerate(titles, start=1)
            if title.casefold() in expected_titles
        ]
        hit = bool(matched_positions)
        if hit:
            hits += 1
            reciprocal_ranks.append(1.0 / matched_positions[0])
        else:
            reciprocal_ranks.append(0.0)
        if not expected_titles and not titles:
            abstention_correct += 1
        precision = (
            sum(1 for title in titles if title.casefold() in expected_titles) / len(titles)
            if titles else (1.0 if not expected_titles else 0.0)
        )
        precision_sum += precision
        results.append({
            "id": str(case.get("id") or index),
            "domain": domain,
            "question": question,
            "expected_titles": sorted(expected_titles),
            "retrieved_titles": titles,
            "hit": hit,
            "reciprocal_rank": round(reciprocal_ranks[-1], 4),
            "correct_abstention": not expected_titles and not titles,
        })

    total = len(rows)
    return {
        "scope": "retrieval_only",
        "cases": total,
        "k": k,
        "hit_rate_at_k": round(hits / total, 4),
        "precision_at_k": round(precision_sum / total, 4),
        "mean_reciprocal_rank": round(sum(reciprocal_ranks) / total, 4),
        "correct_abstentions": abstention_correct,
        "results": results,
        "model_invoked": False,
        "model_training": False,
    }
