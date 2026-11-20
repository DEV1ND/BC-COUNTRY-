"""Build and assess an isolated BPE tokenizer candidate; never touches live weights."""
import json
from pathlib import Path
from bpe_tokenizer import BPETokenizer, load_approved_texts

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "training_extra.json"
OUTPUT = ROOT / "bpe_candidate.json"

def main():
    texts = load_approved_texts(DATA)
    tokenizer = BPETokenizer.train(texts, vocab_size=1024, min_pair_frequency=2)
    before = sum(len(text) for text in texts)
    encoded = [tokenizer.encode(text) for text in texts]
    after = sum(len(ids) for ids in encoded)
    roundtrip = sum(tokenizer.decode(ids) == text for ids, text in zip(encoded, texts))
    payload = tokenizer.to_dict()
    payload["assessment"] = {
        "source": "approved-training-extra",
        "source_texts": len(texts),
        "source_characters": before,
        "encoded_tokens": after,
        "compression_ratio": round(after / max(1, before), 4),
        "roundtrip_passed": roundtrip,
        "roundtrip_total": len(texts),
        "live_model_changed": False,
        "note": "후보 토크나이저만 생성했습니다. 기존 모델 가중치와 서비스 토크나이저는 변경하지 않았습니다."
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["assessment"], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
