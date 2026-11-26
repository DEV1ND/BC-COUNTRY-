"""Integrity audit for isolated Quantum Candidate datasets."""
import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

REQUIRED_EVALUATION_CATEGORIES = {
    "일상 대화", "역사", "과학", "수학", "경제", "사회", "지리", "기술",
    "프로그래밍", "언어", "논리", "복합 추론", "긴 문맥", "사실 확인", "물리", "화학",
}


def normalize_question(value):
    value = unicodedata.normalize("NFC", str(value)).casefold()
    return re.sub(r"[^0-9a-z가-힣]", "", value)


def _load_array(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path}: 최상위 JSON 값은 배열이어야 합니다.")
    return value


def audit_datasets(training_rows, evaluation_rows, feedback_rows):
    if not all(isinstance(rows, list) for rows in (training_rows, evaluation_rows, feedback_rows)):
        raise ValueError("모든 데이터 입력은 JSON 배열이어야 합니다.")
    errors, warnings = [], []

    def inspect(rows, label, required):
        seen, questions = {}, []
        for index, row in enumerate(rows):
            prefix = f"{label}[{index}]"
            if not isinstance(row, dict):
                errors.append(f"{prefix}: 객체가 아닙니다.")
                continue
            for field in required:
                value = row.get(field)
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"{prefix}: {field} 값이 비어 있거나 문자열이 아닙니다.")
            normalized = normalize_question(row.get("question", ""))
            if normalized:
                if normalized in seen:
                    errors.append(f"{prefix}: {label}[{seen[normalized]}]와 질문이 중복됩니다.")
                else:
                    seen[normalized] = index
                questions.append(normalized)
        return questions

    train_q = inspect(training_rows, "training", ("question", "answer", "source"))
    eval_q = inspect(evaluation_rows, "evaluation", ("question", "answer", "category"))

    for i, row in enumerate(training_rows):
        if isinstance(row, dict) and row.get("approved") is not True:
            errors.append(f"training[{i}]: 승인되지 않은 행이 학습 파일에 있습니다.")
    for i, row in enumerate(feedback_rows):
        if not isinstance(row, dict):
            errors.append(f"feedback[{i}]: 객체가 아닙니다.")
            continue
        if row.get("approved") is not True:
            errors.append(f"feedback[{i}]: 승인되지 않은 피드백이 포함되어 있습니다.")
        for field in ("reviewed_by", "correction_source", "source", "question", "answer"):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"feedback[{i}]: {field} 값이 비어 있거나 문자열이 아닙니다.")

    overlap = sorted(set(train_q) & set(eval_q))
    if overlap:
        errors.append(f"학습/독립 평가 질문 중복 {len(overlap)}건: {overlap[:5]}")
    category_counts = Counter(
        str(row.get("category", "")).strip()
        for row in evaluation_rows if isinstance(row, dict)
    )
    missing = sorted(REQUIRED_EVALUATION_CATEGORIES - set(category_counts))
    if missing:
        errors.append("독립 평가 범주 누락: " + ", ".join(missing))
    if len(training_rows) < 10:
        warnings.append("학습 데이터가 10개 미만입니다.")
    if len(evaluation_rows) < 30:
        warnings.append("독립 평가 데이터가 30개 미만입니다.")
    if not feedback_rows:
        warnings.append("사람이 검토한 승인 피드백이 아직 없습니다.")

    return {
        "passed": not errors,
        "training_rows": len(training_rows),
        "evaluation_rows": len(evaluation_rows),
        "approved_feedback_rows": len(feedback_rows),
        "evaluation_category_counts": dict(sorted(category_counts.items())),
        "missing_evaluation_categories": missing,
        "normalized_train_eval_overlap_count": len(overlap),
        "errors": errors,
        "warnings": warnings,
        "scope_warning": "구조·출처·중복·누수 검사이며 사실 정확도나 의미적 정답률을 측정하지 않습니다.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", default="training_extra.json")
    parser.add_argument("--evaluation", default="bpe_independent_eval.json")
    parser.add_argument("--feedback", default="quantum_feedback_approved.json")
    parser.add_argument("--output", default="quantum_dataset_audit_report.json")
    args = parser.parse_args()
    report = audit_datasets(_load_array(args.training), _load_array(args.evaluation), _load_array(args.feedback))
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
