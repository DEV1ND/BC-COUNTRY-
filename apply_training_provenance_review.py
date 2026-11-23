"""Apply a completed human provenance review to a separate dataset copy.

This checks structure and explicit review fields; it cannot prove that claims are true.
It never infers provenance or approves records automatically.
"""
import argparse
import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path

from export_training_provenance_template import SOURCE_FIELDS, _record_id
from training_provenance import audit_rows

TEMPLATE_SCHEMA = "juckex-training-provenance-review-template-v1"
_PLACEHOLDER_VALUES = {"unknown", "n/a", "na", "tbd", "todo", "pending", "curated-v1"}


def _has_real_value(value):
    return isinstance(value, str) and bool(value.strip()) and value.strip().casefold() not in _PLACEHOLDER_VALUES


def _valid_timestamp(value):
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def apply_review(source_rows, template):
    if not isinstance(source_rows, list):
        raise ValueError("원본 학습 데이터는 JSON 배열이어야 합니다.")
    if not isinstance(template, dict) or template.get("schema") != TEMPLATE_SCHEMA:
        raise ValueError("지원하지 않는 provenance review template 형식입니다.")
    records = template.get("records")
    if not isinstance(records, list) or template.get("records_total") != len(records):
        raise ValueError("검토 템플릿 records 또는 records_total 값이 올바르지 않습니다.")
    if "content_included" in template and not isinstance(template["content_included"], bool):
        raise ValueError("content_included 값은 bool이어야 합니다.")

    expected = {}
    for index, row in enumerate(source_rows):
        if not isinstance(row, dict) or row.get("approved") is not True:
            continue
        question, answer = row.get("question"), row.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"원본 승인 데이터 {index}번의 question이 올바르지 않습니다.")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"원본 승인 데이터 {index}번의 answer가 올바르지 않습니다.")
        expected[index] = (_record_id(question, answer), row)

    reviewed = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("검토 레코드는 객체여야 합니다.")
        original_index = record.get("original_index")
        if not isinstance(original_index, int) or isinstance(original_index, bool) or original_index not in expected:
            raise ValueError("원본과 일치하지 않는 original_index가 있습니다.")
        record_id = record.get("record_id")
        expected_id, original = expected[original_index]
        if not isinstance(record_id, str) or record_id != expected_id:
            raise ValueError("원본과 일치하지 않는 record_id가 있습니다.")
        if original_index in reviewed:
            raise ValueError("중복 original_index가 있습니다.")
        for field in ("question", "answer"):
            if field in record and record[field] != original[field]:
                raise ValueError(f"{field} 내용이 원본과 일치하지 않습니다.")

        source = record.get("source")
        if not isinstance(source, dict):
            raise ValueError("source 메타데이터는 객체여야 합니다.")
        for field in SOURCE_FIELDS:
            if not _has_real_value(source.get(field)):
                raise ValueError(f"source.{field}에 실제 검토된 값이 필요합니다.")

        review = record.get("provenance_review")
        if not isinstance(review, dict) or review.get("approved") is not True:
            raise ValueError("사람 검토 승인되지 않은 레코드가 있습니다.")
        reviewer, reviewed_at = review.get("reviewer"), review.get("reviewed_at")
        if not _has_real_value(reviewer):
            raise ValueError("실제 검토자 정보가 필요합니다.")
        if not _valid_timestamp(reviewed_at):
            raise ValueError("reviewed_at은 시간대가 포함된 ISO 8601 날짜·시간이어야 합니다.")
        reviewed[original_index] = (source, {
            "approved": True, "reviewer": reviewer.strip(), "reviewed_at": reviewed_at.strip()
        })

    if set(reviewed) != set(expected):
        missing = len(set(expected) - set(reviewed))
        extra = len(set(reviewed) - set(expected))
        raise ValueError(f"검토 누락 또는 불일치: 누락 {missing}개, 추가 {extra}개")

    output = copy.deepcopy(source_rows)
    for index, (source, review) in reviewed.items():
        output[index]["source_legacy_label"] = output[index].get("source")
        output[index]["source"] = {field: source[field].strip() for field in SOURCE_FIELDS}
        output[index]["provenance_review"] = review

    audit = audit_rows(output)
    if not audit["training_release_eligible"]:
        raise ValueError("적용 후 provenance audit가 학습 적격 판정을 내리지 않았습니다.")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="원본 training_extra.json")
    parser.add_argument("--template", required=True, help="사람이 직접 작성한 검토 템플릿")
    parser.add_argument("--output", required=True, help="별도 출력 파일 경로")
    args = parser.parse_args()
    source_path, output_path = Path(args.input).resolve(), Path(args.output).resolve()
    if output_path == source_path:
        raise SystemExit("원본 데이터 덮어쓰기는 허용되지 않습니다.")
    if output_path.exists():
        raise SystemExit("기존 파일 보호를 위해 이미 존재하는 출력 파일은 덮어쓰지 않습니다.")

    source_rows = json.loads(source_path.read_text(encoding="utf-8"))
    template = json.loads(Path(args.template).read_text(encoding="utf-8"))
    reviewed_rows = apply_review(source_rows, template)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(reviewed_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    report = audit_rows(reviewed_rows)
    print(json.dumps({
        "status": "reviewed_copy_created",
        "output": str(output_path),
        "approved_records": report["approved_records"],
        "metadata_complete_records": report["metadata_complete_records"],
        "training_release_eligible": report["training_release_eligible"],
        "source_dataset_modified": False,
        "human_review_truth_independently_verified": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
