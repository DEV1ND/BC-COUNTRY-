"""Deterministic retrieval benchmark metrics; does not score or train model."""
from collections import defaultdict
import unicodedata
from typing import Callable, Iterable, Mapping

_TITLE_ALIASES = {
    "기후변화": "기후변화",
    "화학원소": "원소(화학)",
    "운영체제": "운영체제",
    "컴퓨터네트워크": "컴퓨터망",
}


def normalize_retrieval_title(value):
    """Normalize whitespace/Unicode and known Korean Wikipedia redirect titles."""
    normalized = "".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())
    return _TITLE_ALIASES.get(normalized, normalized)




def _bounded_integer(value, label, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value


def validate_retrieval_cases(cases: Iterable[Mapping], max_cases: int = 500) -> list[dict]:
    """Validate a fixed benchmark corpus and reject ambiguous or duplicate case IDs."""
    _bounded_integer(max_cases, "max_cases", 1, 500)
    rows = list(cases)
    if not rows:
        raise ValueError("At least one evaluation case is required.")
    if len(rows) > max_cases:
        raise ValueError("Evaluation dataset exceeds the configured case limit.")

    normalized = []
    seen_ids = set()
    for index, case in enumerate(rows):
        if not isinstance(case, Mapping):
            raise ValueError(f"Case {index} must be an object.")
        case_id = case.get("id")
        question = case.get("question")
        domain = case.get("domain")
        expected = case.get("expected_titles")
        if (not isinstance(case_id, str) or not case_id.strip()
                or not isinstance(question, str) or not question.strip()
                or not isinstance(domain, str) or not domain.strip()
                or not isinstance(expected, list)):
            raise ValueError(f"Case {index} requires id, domain, question, and expected_titles.")
        folded_id = case_id.strip().casefold()
        if folded_id in seen_ids:
            raise ValueError(f"Duplicate evaluation case id: {case_id}")
        seen_ids.add(folded_id)

        titles = []
        seen_titles = set()
        for raw_title in expected:
            if not isinstance(raw_title, str) or not raw_title.strip():
                raise ValueError(f"Case {case_id} contains an invalid expected title.")
            title = raw_title.strip()
            normalized_title = normalize_retrieval_title(title)
            if normalized_title in seen_titles:
                raise ValueError(f"Case {case_id} contains duplicate expected titles.")
            seen_titles.add(normalized_title)
            titles.append(title)
        normalized.append({
            "id": case_id.strip(),
            "domain": domain.strip(),
            "question": question.strip(),
            "expected_titles": titles,
        })
    return normalized


def evaluate_retrieval_cases(
    cases: Iterable[Mapping],
    retrieve: Callable[[str, int], list],
    k: int = 5,
) -> dict:
    """Measure expected-title retrieval and safe abstention on a fixed case set."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError("k must be an integer.")
    k = max(1, min(k, 10))
    rows = validate_retrieval_cases(cases)
    results = []
    hits = 0
    reciprocal_ranks = []
    precision_at_k_sum = 0.0
    precision_returned_sum = 0.0
    abstention_correct = 0
    domain_rows = defaultdict(lambda: {
        "cases": 0, "hits": 0, "reciprocal_rank_sum": 0.0,
        "precision_at_k_sum": 0.0, "correct_abstentions": 0,
        "unanswerable_cases": 0, "false_positive_retrievals": 0,
        "irrelevant_retrievals": 0, "retrieved_documents": 0,
    })

    for case in rows:
        question = case["question"]
        domain = case["domain"]
        expected_titles = {normalize_retrieval_title(title) for title in case["expected_titles"]}
        retrieved = retrieve(question, k)
        if not isinstance(retrieved, list):
            raise ValueError(f"Retriever returned a non-list for case {case['id']}.")
        if any(not isinstance(item, Mapping) for item in retrieved):
            raise ValueError(f"Retriever returned a malformed item for case {case['id']}.")
        retrieved = retrieved[:k]
        titles = [item.get("title") for item in retrieved]
        if any(not isinstance(title, str) or not title.strip() for title in titles):
            raise ValueError(f"Retriever returned an empty title or invalid title for case {case['id']}.")
        titles = [title.strip() for title in titles]
        normalized_titles = [normalize_retrieval_title(title) for title in titles]
        if len(normalized_titles) != len(set(normalized_titles)):
            raise ValueError(f"Retriever returned duplicate titles for case {case['id']}.")
        matched_positions = [
            position for position, title in enumerate(titles, start=1)
            if normalize_retrieval_title(title) in expected_titles
        ]
        hit = bool(matched_positions)
        reciprocal_rank = 1.0 / matched_positions[0] if hit else 0.0
        precision_at_k = sum(
            1 for title in titles if normalize_retrieval_title(title) in expected_titles
        ) / k
        precision_returned = (
            sum(1 for title in titles if normalize_retrieval_title(title) in expected_titles) / len(titles)
            if titles else 0.0
        )
        unanswerable = not expected_titles
        correct_abstention = unanswerable and not titles
        false_positive_retrieval = unanswerable and bool(titles)
        relevant_count = sum(1 for title in titles if normalize_retrieval_title(title) in expected_titles)
        irrelevant_count = len(titles) - relevant_count

        hits += int(hit)
        reciprocal_ranks.append(reciprocal_rank)
        precision_at_k_sum += precision_at_k
        precision_returned_sum += precision_returned
        abstention_correct += int(correct_abstention)

        bucket = domain_rows[domain]
        bucket["cases"] += 1
        bucket["hits"] += int(hit)
        bucket["reciprocal_rank_sum"] += reciprocal_rank
        bucket["precision_at_k_sum"] += precision_at_k
        bucket["correct_abstentions"] += int(correct_abstention)
        bucket["unanswerable_cases"] += int(unanswerable)
        bucket["false_positive_retrievals"] += int(false_positive_retrieval)
        bucket["irrelevant_retrievals"] += irrelevant_count
        bucket["retrieved_documents"] += len(titles)

        results.append({
            "id": case["id"],
            "domain": domain,
            "question": question,
            "expected_titles": sorted(expected_titles),
            "retrieved_titles": titles,
            "hit": hit,
            "reciprocal_rank": round(reciprocal_rank, 4),
            "precision_at_k": round(precision_at_k, 4),
            "correct_abstention": correct_abstention,
            "unanswerable": unanswerable,
            "false_positive_retrieval": false_positive_retrieval,
            "relevant_retrieved": relevant_count,
            "irrelevant_retrieved": irrelevant_count,
            "precision_at_returned": round(relevant_count / len(titles), 4) if titles else 0.0,
        })

    total = len(rows)
    domain_metrics = {}
    for domain, values in sorted(domain_rows.items()):
        count = values["cases"]
        domain_metrics[domain] = {
            "cases": count,
            "hit_rate_at_k": round(values["hits"] / count, 4),
            "precision_at_k": round(values["precision_at_k_sum"] / count, 4),
            "mean_reciprocal_rank": round(values["reciprocal_rank_sum"] / count, 4),
            "correct_abstentions": values["correct_abstentions"],
            "unanswerable_cases": values["unanswerable_cases"],
            "false_positive_retrievals": values["false_positive_retrievals"],
            "irrelevant_retrievals": values["irrelevant_retrievals"],
            "retrieved_documents": values["retrieved_documents"],
            "retrieval_noise_rate": (
                round(values["irrelevant_retrievals"] / values["retrieved_documents"], 4)
                if values["retrieved_documents"] else None
            ),
            "safe_abstention_rate": (
                round(values["correct_abstentions"] / values["unanswerable_cases"], 4)
                if values["unanswerable_cases"] else None
            ),
        }

    return {
        "scope": "retrieval_only",
        "cases": total,
        "domains": len(domain_metrics),
        "k": k,
        "hit_rate_at_k": round(hits / total, 4),
        "precision_at_k": round(precision_at_k_sum / total, 4),
        "precision_at_returned": round(precision_returned_sum / total, 4),
        "mean_reciprocal_rank": round(sum(reciprocal_ranks) / total, 4),
        "correct_abstentions": abstention_correct,
        "unanswerable_cases": sum(1 for case in rows if not case["expected_titles"]),
        "false_positive_retrievals": sum(1 for result in results if result["false_positive_retrieval"]),
        "irrelevant_retrievals": sum(result["irrelevant_retrieved"] for result in results),
        "retrieved_documents": sum(len(result["retrieved_titles"]) for result in results),
        "retrieval_noise_rate": (
            round(
                sum(result["irrelevant_retrieved"] for result in results)
                / sum(len(result["retrieved_titles"]) for result in results), 4
            )
            if any(result["retrieved_titles"] for result in results) else None
        ),
        "safe_abstention_rate": (
            round(
                abstention_correct / sum(1 for case in rows if not case["expected_titles"]),
                4,
            )
            if any(not case["expected_titles"] for case in rows) else None
        ),
        "domain_metrics": domain_metrics,
        "results": results,
        "model_invoked": False,
        "model_training": False,
    }
