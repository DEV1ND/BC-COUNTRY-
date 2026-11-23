"""Contract tests for the isolated Quantum Candidate HTTP generation handler.

These tests call the FastAPI endpoint function directly: no server, live secrets,
network access, checkpoint, training, or Production service is required.
"""
import unittest
import sqlite3
import asyncio
import hashlib
from unittest.mock import patch

import torch

from fastapi import HTTPException
from pydantic import ValidationError

import app as quantum_app


class InferenceContractTests(unittest.TestCase):
    def setUp(self):
        self.token_patch = patch.object(quantum_app, "API_TOKEN", "candidate-test-token")
        self.qa_patch = patch.object(quantum_app, "APPROVED_QA_ROWS", [])
        self.model_patch = patch.object(quantum_app, "model", None)
        self.token_patch.start()
        self.qa_patch.start()
        self.model_patch.start()
        self.addCleanup(self.token_patch.stop)
        self.addCleanup(self.qa_patch.stop)
        self.addCleanup(self.model_patch.stop)

    def test_missing_authentication_is_rejected_before_retrieval(self):
        with patch.object(quantum_app, "retrieve_approved_qa") as retrieve:
            with self.assertRaises(HTTPException) as raised:
                quantum_app.generate(quantum_app.GenerateRequest(prompt="안녕하세요"))
        self.assertEqual(raised.exception.status_code, 401)
        retrieve.assert_not_called()

    def test_wrong_bearer_token_is_rejected(self):
        with self.assertRaises(HTTPException) as raised:
            quantum_app.generate(
                quantum_app.GenerateRequest(prompt="안녕하세요"),
                authorization="Bearer wrong-token",
            )
        self.assertEqual(raised.exception.status_code, 401)

    def test_missing_server_token_fails_closed(self):
        with patch.object(quantum_app, "API_TOKEN", ""):
            with self.assertRaises(HTTPException) as raised:
                quantum_app.generate(
                    quantum_app.GenerateRequest(prompt="안녕하세요"),
                    authorization="Bearer candidate-test-token",
                )
        self.assertEqual(raised.exception.status_code, 503)

    def test_reviewed_qa_can_answer_without_loading_a_checkpoint(self):
        match = {
            "matched": True,
            "answer": "검수된 자료에서 확인한 답변입니다.",
            "question": "테스트 질문",
            "score": 1.0,
            "margin": 0.5,
        }
        with patch.object(quantum_app, "retrieve_approved_qa", return_value=match):
            result = quantum_app.generate(
                quantum_app.GenerateRequest(prompt="테스트 질문"),
                authorization="Bearer candidate-test-token",
            )
        self.assertEqual(result["answer"], match["answer"])
        self.assertEqual(result["response_mode"], "approved_qa_retrieval")
        self.assertFalse(result["is_live_inference"])
        self.assertTrue(result["is_live_retrieval"])
        self.assertFalse(result["is_model_generation"])
        self.assertEqual(result["response_mode"], "approved_qa_retrieval")

    def test_no_checkpoint_and_no_retrieval_returns_503_not_fake_answer(self):
        with patch.object(
            quantum_app,
            "retrieve_approved_qa",
            return_value={"matched": False},
        ):
            with self.assertRaises(HTTPException) as raised:
                quantum_app.generate(
                    quantum_app.GenerateRequest(prompt="새로운 질문"),
                    authorization="Bearer candidate-test-token",
                )
        self.assertEqual(raised.exception.status_code, 503)

    def test_prompt_is_trimmed_and_blank_prompt_is_rejected(self):
        self.assertEqual(
            quantum_app.GenerateRequest(prompt="  질문  ").prompt,
            "질문",
        )
        with self.assertRaises(ValidationError):
            quantum_app.GenerateRequest(prompt="  \t\n ")

    def test_prompt_control_characters_are_rejected(self):
        with self.assertRaises(ValidationError):
            quantum_app.GenerateRequest(prompt="질문\x00내용")

    def test_generation_limits_reject_bool_and_float_integers(self):
        for value in (True, 12.5):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    quantum_app.GenerateRequest(prompt="질문", max_new_tokens=value)
                with self.assertRaises(ValidationError):
                    quantum_app.GenerateRequest(prompt="질문", top_k=value)

    def test_generation_limits_enforce_bounds(self):
        for field, value in (
            ("max_new_tokens", 7),
            ("max_new_tokens", 161),
            ("top_k", -1),
            ("top_k", 101),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValidationError):
                    quantum_app.GenerateRequest(prompt="질문", **{field: value})

    def test_non_finite_sampling_configuration_is_rejected(self):
        for field, value in (
            ("temperature", float("nan")),
            ("temperature", float("inf")),
            ("temperature", True),
            ("temperature", "0.7"),
            ("repetition_penalty", float("-inf")),
            ("repetition_penalty", False),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValidationError):
                    quantum_app.GenerateRequest(prompt="질문", **{field: value})

    def test_greedy_generation_rejects_non_finite_or_fully_masked_logits(self):
        from contextlib import ExitStack

        class FakeTokenizer:
            def encode(self, _text):
                return [1]

            def decode(self, _ids):
                return "후보 답변"

        class FakeModel:
            def __init__(self, value):
                self.value = value

            def __call__(self, input_ids):
                return torch.full((1, input_ids.shape[1], 3), self.value)

        for invalid_value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid_value=invalid_value):
                with ExitStack() as stack:
                    stack.enter_context(patch.object(quantum_app, "model", FakeModel(invalid_value)))
                    stack.enter_context(patch.object(quantum_app, "tokenizer", FakeTokenizer()))
                    stack.enter_context(patch.object(
                        quantum_app, "retrieve_approved_qa", return_value={"matched": False}
                    ))
                    stack.enter_context(patch.object(quantum_app, "retrieve_knowledge_diagnostic", return_value={"documents": [], "status": "no_match", "used_fallback": False}))
                    stack.enter_context(patch.object(quantum_app, "build_knowledge_context", return_value=""))
                    stack.enter_context(patch.object(quantum_app, "serialize_knowledge_sources", return_value=[]))
                    with self.assertRaises(HTTPException) as raised:
                        quantum_app.generate(
                            quantum_app.GenerateRequest(prompt="새 질문", temperature=0),
                            authorization="Bearer candidate-test-token",
                        )
                self.assertEqual(raised.exception.status_code, 503)

    def test_public_health_redacts_checkpoint_exception_details(self):
        from unittest.mock import Mock

        connection = Mock()
        connection.execute.return_value.fetchone.return_value = [0]
        from contextlib import ExitStack

        with ExitStack() as stack:
            stack.enter_context(patch.object(
                quantum_app, "knowledge_db", return_value=connection
            ))
            stack.enter_context(patch.object(
                quantum_app, "CHECKPOINT_LOAD_ERROR",
                "OSError: /private/runtime/path/secret-checkpoint.pt",
            ))
            result = quantum_app.health()
        self.assertEqual(result["checkpoint_load_error"], "OSError")
        self.assertNotIn("private", str(result))
        self.assertNotIn("secret-checkpoint", str(result))

    def test_valid_candidate_checkpoint_path_is_labeled_model_generation(self):
        class FakeTokenizer:
            def encode(self, _text):
                return [1]

            def decode(self, _ids):
                return "Quantum 후보 답변입니다."

        class FakeModel:
            def __call__(self, input_ids):
                logits = torch.zeros((1, input_ids.shape[1], 4))
                logits[:, :, 2] = 5.0
                return logits

        from contextlib import ExitStack

        with ExitStack() as stack:
            stack.enter_context(patch.object(quantum_app, "model", FakeModel()))
            stack.enter_context(patch.object(quantum_app, "tokenizer", FakeTokenizer()))
            stack.enter_context(patch.object(
                quantum_app, "retrieve_approved_qa", return_value={"matched": False}
            ))
            stack.enter_context(patch.object(quantum_app, "retrieve_knowledge_diagnostic", return_value={"documents": [], "status": "no_match", "used_fallback": False}))
            stack.enter_context(patch.object(quantum_app, "build_knowledge_context", return_value=""))
            stack.enter_context(patch.object(quantum_app, "serialize_knowledge_sources", return_value=[]))
            result = quantum_app.generate(
                quantum_app.GenerateRequest(prompt="새 질문", max_new_tokens=8),
                authorization="Bearer candidate-test-token",
            )

        self.assertEqual(result["response_mode"], "model_generation")
        self.assertTrue(result["is_live_inference"])
        self.assertFalse(result["is_live_retrieval"])
        self.assertTrue(result["is_model_generation"])
        self.assertEqual(result["answer"], "Quantum 후보 답변입니다.")
        self.assertEqual(result["generated_tokens"], 8)
        self.assertEqual(result["generation_diagnostics"]["generated_tokens"], 8)
        self.assertIn("interpretation_limit", result["generation_diagnostics"])
        self.assertIn("토큰 진단은 문장 품질 평가가 아닙니다.", result["quality_warning"])
        self.assertEqual(result["generation_stop_reason"], "max_new_tokens")

    def test_public_wikipedia_seed_summary_redacts_exception_and_document_details(self):
        result = quantum_app.public_wikipedia_seed_status({
            "status": "failed",
            "error": "OSError: /private/runtime/secret.db",
            "imported_this_start": 0,
            "page_diagnostics": [{"title": "private draft", "url": "https://internal.invalid"}],
        })
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_type"], "OSError")
        self.assertEqual(result["imported_this_start"], 0)
        self.assertNotIn("error", result)
        self.assertNotIn("page_diagnostics", result)
        self.assertNotIn("private", str(result))


    def test_malformed_model_output_tensor_is_rejected_without_leaking_details(self):
        class FakeTokenizer:
            vocab = ["<unk>", "가", "나"]

            def encode(self, _text):
                return [1]

            def decode(self, _ids):
                return "후보 답변"

        malformed_outputs = (
            torch.zeros((3,)),
            torch.zeros((1, 3)),
            torch.zeros((2, 1, 3)),
            torch.zeros((1, 2, 3)),
            torch.zeros((1, 1, 4)),
            {"logits": torch.zeros((1, 1, 3))},
        )

        class FakeModel:
            def __init__(self, output):
                self.output = output

            def __call__(self, _input_ids):
                return self.output

        from contextlib import ExitStack
        for output in malformed_outputs:
            with self.subTest(output_type=type(output).__name__, shape=getattr(output, "shape", None)):
                with ExitStack() as stack:
                    stack.enter_context(patch.object(quantum_app, "model", FakeModel(output)))
                    stack.enter_context(patch.object(quantum_app, "tokenizer", FakeTokenizer()))
                    stack.enter_context(patch.object(
                        quantum_app, "retrieve_approved_qa", return_value={"matched": False}
                    ))
                    stack.enter_context(patch.object(quantum_app, "retrieve_knowledge_diagnostic", return_value={"documents": [], "status": "no_match", "used_fallback": False}))
                    stack.enter_context(patch.object(quantum_app, "build_knowledge_context", return_value=""))
                    stack.enter_context(patch.object(quantum_app, "serialize_knowledge_sources", return_value=[]))
                    with self.assertRaises(HTTPException) as raised:
                        quantum_app.generate(
                            quantum_app.GenerateRequest(prompt="새 질문", max_new_tokens=8),
                            authorization="Bearer candidate-test-token",
                        )
                self.assertEqual(raised.exception.status_code, 503)
                self.assertEqual(
                    raised.exception.detail,
                    "Quantum 후보 모델이 올바른 출력 형식을 반환하지 않았습니다.",
                )

    def test_model_runtime_failure_returns_generic_service_error(self):
        class FakeTokenizer:
            vocab = ["<unk>", "가", "나"]

            def encode(self, _text):
                return [1]

            def decode(self, _ids):
                return "후보 답변"

        class BrokenModel:
            def __call__(self, _input_ids):
                raise RuntimeError("private checkpoint implementation detail")

        from contextlib import ExitStack
        with ExitStack() as stack:
            stack.enter_context(patch.object(quantum_app, "model", BrokenModel()))
            stack.enter_context(patch.object(quantum_app, "tokenizer", FakeTokenizer()))
            stack.enter_context(patch.object(
                quantum_app, "retrieve_approved_qa", return_value={"matched": False}
            ))
            stack.enter_context(patch.object(quantum_app, "retrieve_knowledge_diagnostic", return_value={"documents": [], "status": "no_match", "used_fallback": False}))
            stack.enter_context(patch.object(quantum_app, "build_knowledge_context", return_value=""))
            stack.enter_context(patch.object(quantum_app, "serialize_knowledge_sources", return_value=[]))
            with self.assertRaises(HTTPException) as raised:
                quantum_app.generate(
                    quantum_app.GenerateRequest(prompt="새 질문", max_new_tokens=8),
                    authorization="Bearer candidate-test-token",
                )
        self.assertEqual(raised.exception.status_code, 503)
        self.assertNotIn("private checkpoint", raised.exception.detail)


    def test_generation_reports_separator_stop_reason(self):
        class FakeTokenizer:
            vocab = ["<unk>", "가", "나"]

            def encode(self, _text):
                return [1]

            def decode(self, _ids):
                return "답변입니다.\n\n"

        class FakeModel:
            def __call__(self, input_ids):
                logits = torch.zeros((1, input_ids.shape[1], 3))
                logits[:, :, 2] = 5.0
                return logits

        from contextlib import ExitStack
        with ExitStack() as stack:
            stack.enter_context(patch.object(quantum_app, "model", FakeModel()))
            stack.enter_context(patch.object(quantum_app, "tokenizer", FakeTokenizer()))
            stack.enter_context(patch.object(
                quantum_app, "retrieve_approved_qa", return_value={"matched": False}
            ))
            stack.enter_context(patch.object(quantum_app, "retrieve_knowledge_diagnostic", return_value={"documents": [], "status": "no_match", "used_fallback": False}))
            stack.enter_context(patch.object(quantum_app, "build_knowledge_context", return_value=""))
            stack.enter_context(patch.object(quantum_app, "serialize_knowledge_sources", return_value=[]))
            result = quantum_app.generate(
                quantum_app.GenerateRequest(prompt="새 질문", max_new_tokens=20),
                authorization="Bearer candidate-test-token",
            )
        self.assertEqual(result["answer"], "답변입니다.")
        self.assertEqual(result["generated_tokens"], 6)
        self.assertEqual(result["generation_stop_reason"], "separator")


    def test_sqlite_connection_failure_is_reported_and_model_generation_continues(self):
        class FakeTokenizer:
            vocab = ["<unk>", "가", "나"]

            def encode(self, _text):
                return [1]

            def decode(self, _ids):
                return "검색 장애와 무관한 후보 답변입니다."

        class FakeModel:
            def __call__(self, input_ids):
                logits = torch.zeros((1, input_ids.shape[1], 3))
                logits[:, :, 2] = 5.0
                return logits

        from contextlib import ExitStack
        with ExitStack() as stack:
            stack.enter_context(patch.object(quantum_app, "model", FakeModel()))
            stack.enter_context(patch.object(quantum_app, "tokenizer", FakeTokenizer()))
            stack.enter_context(patch.object(
                quantum_app, "retrieve_approved_qa", return_value={"matched": False}
            ))
            stack.enter_context(patch.object(
                quantum_app, "knowledge_db", side_effect=sqlite3.OperationalError("private db path")
            ))
            result = quantum_app.generate(
                quantum_app.GenerateRequest(prompt="새 질문", max_new_tokens=8),
                authorization="Bearer candidate-test-token",
            )

        self.assertEqual(result["response_mode"], "model_generation")
        self.assertEqual(result["answer"], "검색 장애와 무관한 후보 답변입니다.")
        self.assertEqual(result["knowledge_retrieval"], {
            "status": "search_unavailable",
            "used_fallback": False,
            "source_count": 0,
            "rejected_source_rows": 0,
            "rejected_integrity_rows": 0,
        })


    def _valid_import_document(self):
        body = ("검증된 위키백과 문서의 본문 예시입니다. " * 12).strip()
        title = "인공지능"
        url = "https://ko.wikipedia.org/wiki/인공지능"
        return {
            "page_id": "12345",
            "title": title,
            "url": url,
            "text": body,
            "license": "CC BY-SA 4.0",
            "attribution": (
                f"Wikipedia contributors, {title}, {url}; licensed under "
                "CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)."
            ),
            "revision_id": "987654",
            "content_hash": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        }

    def test_knowledge_import_commits_canonical_and_fts_rows_together(self):
        class FakeRequest:
            async def json(self):
                return {"documents": [self._document]}

        request = FakeRequest()
        request._document = self._valid_import_document()
        connection = sqlite3.connect(":memory:")
        connection.execute("""
            CREATE TABLE knowledge_docs(
                page_id TEXT PRIMARY KEY, title TEXT, url TEXT, text TEXT,
                license TEXT, attribution TEXT, revision_id TEXT, content_hash TEXT,
                imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.execute("""
            CREATE TABLE knowledge_fts(
                page_id TEXT, title TEXT, text TEXT
            )
        """)

        class TrackingConnection:
            def __init__(self, inner):
                self.inner = inner
                self.closed = False

            def execute(self, statement, params=()):
                return self.inner.execute(statement, params)

            def commit(self):
                return self.inner.commit()

            def rollback(self):
                return self.inner.rollback()

            def close(self):
                self.closed = True

            @property
            def in_transaction(self):
                return self.inner.in_transaction

        tracked = TrackingConnection(connection)
        with patch.object(quantum_app, "require_admin"):
            with patch.object(quantum_app, "knowledge_db", return_value=tracked):
                result = asyncio.run(quantum_app.import_knowledge(
                    request, authorization="Bearer candidate-test-token"
                ))

        self.assertTrue(tracked.closed)
        self.assertTrue(result["ok"])
        self.assertEqual(result["imported"], 1)
        canonical = connection.execute(
            "SELECT page_id,title,revision_id,content_hash FROM knowledge_docs"
        ).fetchone()
        indexed = connection.execute(
            "SELECT page_id,title,text FROM knowledge_fts"
        ).fetchone()
        self.assertEqual(canonical[0], request._document["page_id"])
        self.assertEqual(canonical[1], request._document["title"])
        self.assertEqual(canonical[2], request._document["revision_id"])
        self.assertEqual(canonical[3], request._document["content_hash"])
        self.assertEqual(indexed[0], canonical[0])
        self.assertEqual(indexed[1], canonical[1])
        self.assertEqual(indexed[2], request._document["text"].strip())
        self.assertEqual(connection.in_transaction, False)
        connection.close()

    def test_knowledge_import_rejects_malformed_batch_before_opening_database(self):
        class FakeRequest:
            async def json(self):
                document = self._document.copy()
                document["page_id"] = 12345
                return {"documents": [document]}

        request = FakeRequest()
        request._document = self._valid_import_document()
        with patch.object(quantum_app, "require_admin"):
            with patch.object(quantum_app, "knowledge_db") as open_db:
                with self.assertRaises(HTTPException) as raised:
                    asyncio.run(quantum_app.import_knowledge(
                        request, authorization="Bearer candidate-test-token"
                    ))
        self.assertEqual(raised.exception.status_code, 422)
        open_db.assert_not_called()

    def test_knowledge_import_rolls_back_all_writes_when_fts_insert_fails(self):
        class FakeRequest:
            async def json(self):
                return {"documents": [self._document]}

        class FakeConnection:
            def __init__(self):
                self.calls = []
                self.rolled_back = False
                self.committed = False
                self.closed = False

            def execute(self, statement, params=()):
                self.calls.append(statement)
                if "INSERT INTO knowledge_fts" in statement:
                    raise sqlite3.OperationalError("simulated FTS write failure")
                return self

            def rollback(self):
                self.rolled_back = True

            def commit(self):
                self.committed = True

            def close(self):
                self.closed = True

        request = FakeRequest()
        request._document = self._valid_import_document()
        connection = FakeConnection()
        with patch.object(quantum_app, "require_admin"):
            with patch.object(quantum_app, "knowledge_db", return_value=connection):
                with self.assertRaises(HTTPException) as raised:
                    asyncio.run(quantum_app.import_knowledge(
                        request, authorization="Bearer candidate-test-token"
                    ))
        self.assertEqual(raised.exception.status_code, 503)
        self.assertTrue(connection.rolled_back)
        self.assertFalse(connection.committed)
        self.assertTrue(connection.closed)


if __name__ == "__main__":
    unittest.main()
