"""Export a blank, human-editable provenance review template.

This tool does not infer sources, licenses, permission, attribution, or approvals.
It never edits the training dataset.
"""
import argparse
import hashlib
import json
from pathlib import Path

SOURCE_FIELDS = ("kind", "reference", "license", "attribution", "permission_basis")


def export_template(rows):
    if not isinstance(rows, list):
        raise ValueError("학습 데이터 최상위 구조는 JSON 배열이어야 합니다.")
    records = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("approved") is not True:
            continue
        question = str(row.get("question", ""))
        answer = str(row.get("answer", ""))
        row_id = hashlib.sha256((question + "\\0" + answer).encode("utf-8")).hexdigest()
        legacy_source = row.get("source")
        records.append({
            "record_id": row_id,
            "original_index": index,
            "question": question,
            "answer": answer,
            "legacy_source_label": legacy_source if isinstance(legacy_source, str) else None,
            "source": {field: "" for field in SOURCE_FIELDS},
            "provenance_review": {
                "approved": False,
                "reviewer": "",
                "reviewed_at": "",
            },
        })
    return {
        "schema": "juckex-training-provenance-review-template-v1",
        "records_total": len(records),
        "warning": "빈 양식입니다. 실제 근거를 확인한 검토자만 출처·라이선스·권한과 검토 결과를 기입하세요. 자동으로 승인되지 않습니다.",
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="training_extra.json")
    parser.add_argument("--output", default="training_provenance_review_template.json")
    args = parser.parse_args()
    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    template = export_template(rows)
    Path(args.output).write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "blank_review_template_created",
        "records": template["records_total"],
        "output": args.output,
        "dataset_modified": False,
        "automatic_approval": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
