"""Regression tests for the isolated BPE candidate data pipeline."""
import unittest

from bpe_candidate_train import augment_training_rows, build_candidate_tokenizer, clean_pairs, split_pairs
from bpe_tokenizer import encode_prompt_answer


class CandidateDataPipelineTests(unittest.TestCase):
    def test_clean_pairs_requires_approval_and_removes_duplicates(self):
        rows = [
            {"approved": True, "question": "질문 A", "answer": "충분히 긴 답변입니다."},
            {"approved": False, "question": "질문 B", "answer": "승인되지 않은 답변입니다."},
            {"approved": True, "question": "질문 A", "answer": "충분히 긴 답변입니다."},
        ]
        cleaned = clean_pairs(rows)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["question"], "질문 A")

    def test_split_keeps_question_groups_disjoint(self):
        rows = clean_pairs([
            {"approved": True, "question": f"질문 {i}", "answer": f"답변 내용 {i} 충분히 길게"}
            for i in range(20)
        ])
        train, valid = split_pairs(rows)
        train_questions = {row["question"].casefold() for row in train}
        valid_questions = {row["question"].casefold() for row in valid}
        self.assertTrue(train_questions)
        self.assertTrue(valid_questions)
        self.assertTrue(train_questions.isdisjoint(valid_questions))

    def test_augmentation_uses_training_rows_only(self):
        train = [{"question": "훈련 질문", "answer": "훈련 답변"}]
        augmented = augment_training_rows(train)
        self.assertEqual(len(augmented), 4)
        self.assertEqual({row["answer"] for row in augmented}, {"훈련 답변"})
        self.assertTrue(all("평가" not in row["question"] for row in augmented))

    def test_tokenizer_round_trip_and_prompt_boundary(self):
        train_rows = [{"text": "질문:오늘 날씨\n답변:맑습니다"}]
        validation_rows = [{"text": "검증 문자 🧬"}]
        tokenizer = build_candidate_tokenizer(train_rows, validation_rows, vocab_size=64)
        prompt = "질문:오늘"
        answer = "맑습니다"
        ids, prompt_length = encode_prompt_answer(prompt, answer, tokenizer)
        self.assertEqual(tokenizer.decode(ids), prompt + answer + "\n\n")
        self.assertEqual(prompt_length, len(tokenizer.encode(prompt)))
        self.assertIn("🧬", tokenizer.initial_symbols)
        self.assertNotIn("🧪", tokenizer.initial_symbols)
        self.assertNotIn("☃", tokenizer.initial_symbols)


if __name__ == "__main__":
    unittest.main()
