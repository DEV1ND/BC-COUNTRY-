"""Small repeatable quality checks for retrieval and the isolated BPE candidate."""
import json
import unittest
from pathlib import Path
from app import _curated_exact_answer, _curated_similar_answer
from bpe_tokenizer import BPETokenizer, load_approved_texts, encode_prompt_answer

ROOT = Path(__file__).resolve().parent

class RetrievalBenchmark(unittest.TestCase):
    def test_exact_curated_questions(self):
        rows = json.loads((ROOT / "training_extra.json").read_text(encoding="utf-8"))
        approved = [r for r in rows if isinstance(r, dict) and r.get("approved") is True]
        sample = approved[:30]
        hits = sum(bool(_curated_exact_answer(row["question"])) for row in sample)
        self.assertGreaterEqual(hits, max(1, int(len(sample) * 0.95)))

    def test_known_paraphrases_retrieve_relevant_material(self):
        cases = [
            ("AI가 뭐야?", ("인공지능", "ai")),
            ("파이썬이 뭐야?", ("파이썬", "python")),
            ("HTML이 뭐야?", ("HTML", "웹")),
            ("삼권분립이 뭐야?", ("삼권분립", "권력")),
        ]
        for question, expected_terms in cases:
            answer = _curated_similar_answer(question)
            self.assertTrue(answer, msg=f"검색 결과 없음: {question}")
            self.assertTrue(any(term.casefold() in answer.casefold() for term in expected_terms),
                            msg=f"관련 근거 부족: {question} -> {answer[:120]}")

    def test_unrelated_question_does_not_force_curated_answer(self):
        answer = _curated_similar_answer("은하철도 우주선 엔진의 양자 추진 원리는?")
        self.assertEqual(answer, "")

class BPETokenizerCandidate(unittest.TestCase):
    def test_validation_characters_do_not_break_roundtrip(self):
        train_texts = ["질문:인공지능이란 무엇인가? 답변:컴퓨터가 지능적인 작업을 수행하도록 만드는 기술입니다."]
        validation_texts = ["질문:양자컴퓨팅은 무엇인가? 답변:양자 상태를 이용해 계산하는 방식입니다."]
        tokenizer = BPETokenizer.train(
            train_texts,
            vocab_size=256,
            min_pair_frequency=2,
            extra_alphabet_texts=validation_texts,
        )
        for text in validation_texts:
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)


    def test_prompt_answer_boundary_is_not_merged(self):
        prompt = "질문:가\n답변:"
        answer = "나라는 독립된 답변입니다."
        tokenizer = BPETokenizer.train(
            [prompt + answer + "\n\n"],
            vocab_size=256,
            min_pair_frequency=1,
        )
        combined, prompt_length = encode_prompt_answer(prompt, answer, tokenizer)
        self.assertEqual(combined[:prompt_length], tokenizer.encode(prompt))
        self.assertEqual(combined[prompt_length:], tokenizer.encode(answer + "\n\n"))
        self.assertEqual(tokenizer.decode(combined), prompt + answer + "\n\n")

    def test_roundtrip_and_unknown_handling(self):
        texts = load_approved_texts(ROOT / "training_extra.json")
        tokenizer = BPETokenizer.train(texts, vocab_size=512, min_pair_frequency=2)
        for text in texts[:50]:
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertIn(0, tokenizer.encode("새로운🛰문자"))

if __name__ == "__main__":
    unittest.main()
