"""Build-time smoke checks for the isolated Genesis candidate."""
import app

assert app.model is not None, f"Genesis checkpoint failed to load: {app.MODEL_ERROR}"
assert app.tokenizer is not None
assert app.PARAMS == 248576, f"Unexpected parameter count: {app.PARAMS}"
assert app.MODEL_STEPS == 1700, f"Unexpected checkpoint step count: {app.MODEL_STEPS}"
assert len(app.QA) >= 1, "No approved Q&A loaded"

for sample in ("인공지능이란 무엇인가요?", "질문:알고리즘이란 무엇인가요?\n답변:"):
    ids = app.tokenizer.encode(sample)
    restored = app.tokenizer.decode(ids)
    assert restored == sample, f"BPE round trip failed for: {sample!r}"

hit = app.retrieve("인공지능이란 무엇인가요?")
assert hit and hit["score"] == 1.0, "Exact curated-Q&A retrieval failed"

generated = app.generate("태양계에서 가장 큰 행성은?", 24)
assert isinstance(generated, str), "Generation did not return text"

print({
    "status": "smoke_tests_passed",
    "parameters": app.PARAMS,
    "training_steps": app.MODEL_STEPS,
    "approved_qa_count": len(app.QA),
    "tokenizer_roundtrip": "passed",
    "exact_retrieval": "passed",
    "own_model_generation": "returned",
    "semantic_quality": "not_asserted_by_smoke_test",
})
