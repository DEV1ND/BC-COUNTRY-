"""Exercise the real FastAPI generation handler against the loaded local checkpoint."""
import json
import os

os.environ.setdefault("QUANTUM_CHECKPOINT_PATH", "bpe_candidate_checkpoint.pt")
from app import API_TOKEN, GenerateRequest, generate, model

if model is None:
    raise SystemExit("CHECKPOINT_NOT_LOADED")

authorization = f"Bearer {API_TOKEN}" if API_TOKEN else None
questions = [
    "형광펜이 뭐야?",
    "태양계에서 가장 큰 행성은?",
    "판 구조론이란 무엇인가요?",
    "오늘 날씨 어때?",
]
for question in questions:
    result = generate(
        GenerateRequest(prompt=question, max_new_tokens=64, temperature=0.0),
        authorization=authorization,
    )
    if result.get("response_mode") != "transformer_generation":
        raise SystemExit(f"NOT_TRANSFORMER_GENERATION: {result}")
    if result.get("generated_token_count", 0) <= 0 or not result.get("answer"):
        raise SystemExit(f"EMPTY_GENERATION: {result}")
    print(json.dumps({
        "question": question,
        "model": result.get("model"),
        "response_mode": result.get("response_mode"),
        "generated_token_count": result.get("generated_token_count"),
        "answer": result.get("answer"),
        "knowledge_sources": result.get("knowledge_sources", []),
    }, ensure_ascii=False))
