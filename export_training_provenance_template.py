"""Export a blank, human-editable provenance review template.

This tool does not infer sources, licenses, permission, attribution, or approvals.
It never edits the training dataset. Content can be omitted from shareable artifacts.
"""
import argparse
import hashlib
import json
from pathlib import Path

SOURCE_FIELDS = ("kind", "reference", "license", "attribution", "permission_basis")


def export_template(rows, include_content=True):
    if not isinstance(rows, list):
        raise ValueError("학습 데이터 최상위 구조는 JSON 배열이어야 합니다.")
    if not isinstance(include_content, bool):
        raise ValueError("include_content는 bool이어야 합니다.")
    records = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("approved") is not True:
            continue
        question = str(row.get("question", ""))
        answer = str(row.get("answer", ""))
        row_id = hashlib.sha256((question + "\\0" + answer).encode("utf-8")).hexdigest()
        legacy_source = row.get("source")
        record = {
            "record_id": row_id,
            "original_index": index,
            "source": {field: "" for field in SOURCE_FIELDS},
            "provenance_review": {
                "approved": False,
                "reviewer": "",
                "reviewed_at": "",
            },
        }
        if include_content:
            record["question"] = question
            record["answer"] = answer
            record["legacy_source_label"] = legacy_source if isinstance(legacy_source, str) else None
        records.append(record)
    return {
        "schema": "juckex-training-provenance-review-template-v1",
        "records_total": len(records),
        "content_included": include_content,
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
    parser.add_argument(
        "--omit-content",
        action="store_true",
        help="질문·답변·기존 출처 라벨을 템플릿에서 제외하고 해시 식별자만 내보냅니다.",
    )
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
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
