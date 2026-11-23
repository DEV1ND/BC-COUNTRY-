"""Export a blank, human-editable provenance review template.

The stable SHA-256 record identifier is pseudonymous, not anonymization: low-entropy
question/answer pairs may be guessable. Do not publish this template publicly.
This tool never infers provenance or edits the training dataset.
"""
import argparse
import hashlib
import json
from pathlib import Path

SOURCE_FIELDS = ("kind", "reference", "license", "attribution", "permission_basis")


def _record_id(question, answer):
    return hashlib.sha256((question + "\0" + answer).encode("utf-8")).hexdigest()


def export_template(rows, include_content=True):
    if not isinstance(rows, list):
        raise ValueError("학습 데이터 최상위 구조는 JSON 배열이어야 합니다.")
    if not isinstance(include_content, bool):
        raise ValueError("include_content는 bool이어야 합니다.")
    records = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("approved") is not True:
            continue
        question = row.get("question")
        answer = row.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"승인 데이터 {index}번의 question은 비어 있지 않은 문자열이어야 합니다.")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"승인 데이터 {index}번의 answer는 비어 있지 않은 문자열이어야 합니다.")
        record = {
            "record_id": _record_id(question, answer),
            "original_index": index,
            "source": {field: "" for field in SOURCE_FIELDS},
            "provenance_review": {"approved": False, "reviewer": "", "reviewed_at": ""},
        }
        if include_content:
            record["question"] = question
            record["answer"] = answer
            legacy_source = row.get("source")
            record["legacy_source_label"] = legacy_source if isinstance(legacy_source, str) else None
        records.append(record)
    return {
        "schema": "juckex-training-provenance-review-template-v1",
        "records_total": len(records),
        "content_included": include_content,
        "identifier_notice": (
            "record_id는 원본 대조용 결정적 해시 식별자이며 익명화가 아닙니다. "
            "추측 가능한 질문·답변은 해시로 추측될 수 있으므로 템플릿을 공개 저장소에 올리지 마세요."
        ),
        "warning": (
            "빈 양식입니다. 실제 근거를 확인한 검토자만 출처·라이선스·권한과 검토 결과를 기입하세요. "
            "자동으로 승인되지 않습니다. content_included=false이면 원본 데이터와 record_id로 대조하세요."
        ),
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="training_extra.json")
    parser.add_argument("--output", default="training_provenance_review_template.json")
    parser.add_argument("--omit-content", action="store_true",
                        help="질문·답변·기존 출처 라벨을 제외한 해시 식별자 양식을 만듭니다.")
    args = parser.parse_args()
    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    template = export_template(rows, include_content=not args.omit_content)
    Path(args.output).write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "blank_review_template_created",
        "records": template["records_total"],
        "output": args.output,
        "content_included": template["content_included"],
        "dataset_modified": False,
        "automatic_approval": False,
        "identifier_is_anonymized": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
