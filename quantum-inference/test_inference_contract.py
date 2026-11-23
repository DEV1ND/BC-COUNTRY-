"""Contract tests for the isolated Quantum Candidate HTTP generation handler.

These tests call the FastAPI endpoint function directly: no server, live secrets,
network access, checkpoint, training, or Production service is required.
"""
import unittest
from unittest.mock import patch

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
        self.assertTrue(result["is_live_inference"])

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


if __name__ == "__main__":
    unittest.main()
