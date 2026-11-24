"""Build an isolated BPE tokenizer only from provenance-released data.

Tokenizer construction learns merge statistics from text and is therefore treated
as a data-learning operation. It must pass the same provenance and reviewer identity
release gate as model training. The CLI has no identity verifier wired yet and fails
closed until that integration exists.
"""
import argparse
import json
from pathlib import Path

from bpe_tokenizer import BPETokenizer
from training_provenance import audit_rows

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "training_extra.json"
OUTPUT = ROOT / "bpe_candidate.json"


def build_candidate(rows, identity_verifier=None, vocab_size=1024):
    provenance = audit_rows(rows, identity_verifier=identity_verifier)
    if not provenance["training_release_eligible"]:
        raise RuntimeError(
            "BPE 토크나이저 후보 생성 중단: 출처·라이선스·사용 권한 메타데이터와 "
            "신뢰된 검토자 신원 인증을 먼저 완료해야 합니다."
        )

    texts = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("approved") is not True:
            continue
        question, answer = row.get("question"), row.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            raise ValueError(f"승인 데이터 {index}의 질문과 답변은 문자열이어야 합니다.")
        question, answer = question.strip(), answer.strip()
        if len(question) < 2 or len(answer) < 8:
            raise ValueError(f"승인 데이터 {index}의 질문 또는 답변이 너무 짧습니다.")
        texts.extend([question, answer])
    if not texts:
        raise RuntimeError("BPE 토크나이저를 만들 승인된 텍스트가 없습니다.")

    tokenizer = BPETokenizer.train(texts, vocab_size=vocab_size, min_pair_frequency=2)
    encoded = [tokenizer.encode(text) for text in texts]
    roundtrip = sum(tokenizer.decode(ids) == text for ids, text in zip(encoded, texts))
    payload = tokenizer.to_dict()
    payload["assessment"] = {
        "source": "provenance-released-training-extra",
        "source_texts": len(texts),
        "source_characters": sum(len(text) for text in texts),
        "encoded_tokens": sum(len(ids) for ids in encoded),
        "compression_ratio": round(sum(len(ids) for ids in encoded) / max(
            1, sum(len(text) for text in texts)
        ), 4),
        "roundtrip_passed": roundtrip,
        "roundtrip_total": len(texts),
        "metadata_complete_records": provenance["metadata_complete_records"],
        "reviewer_identity_verified_records": provenance["reviewer_identity_verified_records"],
        "training_release_eligible": provenance["training_release_eligible"],
        "identity_verifier_configured": provenance["identity_verifier_configured"],
        "live_model_changed": False,
        "note": "격리된 후보 토크나이저입니다. 운영 가중치와 서비스 토크나이저는 변경하지 않았습니다.",
    }
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DATA))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    # No verifier is configured in this CLI; provenance release remains fail-closed.
    payload = build_candidate(rows)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["assessment"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
