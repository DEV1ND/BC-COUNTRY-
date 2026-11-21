"""Build-time checks for model loading, retrieval, and the FastAPI routes."""
import app
from fastapi.testclient import TestClient

assert app.model is not None, f"Genesis checkpoint failed to load: {app.MODEL_ERROR}"
assert app.tokenizer is not None
assert app.PARAMS == 248576, f"Unexpected parameter count: {app.PARAMS}"
assert app.MODEL_STEPS == 1700, f"Unexpected checkpoint step count: {app.MODEL_STEPS}"
assert len(app.QA) >= 1, "No approved Q&A loaded"

for sample in ("인공지능이란 무엇인가요?", "질문:알고리즘이란 무엇인가요?\n답변:"):
    ids = app.tokenizer.encode(sample)
    assert app.tokenizer.decode(ids) == sample, f"BPE round trip failed: {sample!r}"

hit = app.retrieve("인공지능이란 무엇인가요?")
assert hit and hit["score"] == 1.0, "Exact curated-Q&A retrieval failed"

client = TestClient(app.app)
root = client.get("/")
assert root.status_code == 200 and root.json()["service"] == "JUCKEX Genesis Beta"
health = client.get("/health")
assert health.status_code == 200 and health.json()["model_loaded"] is True
ready = client.get("/ready")
assert ready.status_code == 200 and ready.json()["ready"] is True
model_info = client.get("/v1/model")
assert model_info.status_code == 200 and model_info.json()["parameters"] == 248576

retrieval_response = client.post("/v1/generate", json={"question":"인공지능이란 무엇인가요?"})
assert retrieval_response.status_code == 200
assert retrieval_response.json()["mode"] == "knowledge_match"
assert retrieval_response.json()["answer"]

generation_response = client.post("/v1/generate", json={
    "question":"태양계에서 가장 큰 행성은?", "max_new_tokens":24
})
assert generation_response.status_code == 200
assert generation_response.json()["mode"] == "model_generation"
assert generation_response.json()["answer"]

invalid_response = client.post("/v1/generate", json={"question":""})
assert invalid_response.status_code == 422

print({
    "status": "api_integration_smoke_tests_passed",
    "parameters": app.PARAMS,
    "training_steps": app.MODEL_STEPS,
    "approved_qa_count": len(app.QA),
    "tokenizer_roundtrip": "passed",
    "health_and_readiness": "passed",
    "exact_retrieval_api": "passed",
    "own_model_generation_api": "passed",
    "input_validation": "passed",
    "semantic_quality": "not_asserted_by_smoke_test",
})
