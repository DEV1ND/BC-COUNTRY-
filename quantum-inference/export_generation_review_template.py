"""Export a blank, case-aligned human review template for a Candidate report."""
import argparse
import json
from pathlib import Path


def build_review_template(report):
    if not isinstance(report, dict):
        raise ValueError("Generation report must be a JSON object.")
    if report.get("training_eligible") is not False:
        raise ValueError("Generation report must be training_eligible=false.")
    if report.get("human_review_required") is not True:
        raise ValueError("Generation report must require human review.")
    results = report.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("Generation report must contain results.")
    ids = []
    reviews = []
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            raise ValueError("Generation result " + str(index) + " must be an object.")
        case_id = result.get("id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in ids:
            raise ValueError("Generation results must have unique non-empty string IDs.")
        ids.append(case_id)
        reviews.append({
            "id": case_id,
            "reviewer": "",
            "reviewed_at": "",
            "correctness": None,
            "relevance": None,
            "completeness": None,
            "korean_naturalness": None,
            "uncertainty_handling": None,
            "hallucination_detected": None,
            "notes": "",
            "evidence_refs": [],
        })
    return {
        "benchmark_id": report.get("benchmark_id"),
        "instructions": "사람이 직접 답변을 검토하고 각 항목을 1~5로 평가하세요. 환각 여부는 true/false로 기록하세요. 사실 확인 항목은 근거 URL 또는 문서 식별자를 evidence_refs에 넣으세요. 빈 값은 검토 완료로 인정되지 않습니다.",
        "rating_scale": {
            "1": "매우 부족",
            "2": "부족",
            "3": "보통",
            "4": "좋음",
            "5": "매우 좋음",
        },
        "reviews": reviews,
        "automatic_grading": False,
        "training_eligible": False,
    }


def main():
    parser = argparse.ArgumentParser(description="Export a blank human review template.")
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    template = build_review_template(report)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing review template.")
    output.write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "blank_human_review_template_created",
        "cases": len(template["reviews"]),
        "automatic_grading": False,
        "training_eligible": False,
        "output": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
