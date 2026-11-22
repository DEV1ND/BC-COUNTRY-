import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from wikipedia_pretrain import load_corpus, split_documents


def row(page_id, status="approved", note="MANUAL_APPROVAL: reviewed against source"):
    text = ("한국어 문서 내용입니다. " * 20).strip()
    return {
        "page_id": page_id, "title": "테스트 문서", 
        "url": "https://ko.wikipedia.org/wiki/테스트",
        "revision_id": "12345",
        "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text": text, "license": "CC BY-SA 4.0",
        "attribution": "위키백과 contributors", "status": status,
        "reviewer_note": note,
    }


class CorpusValidationTests(unittest.TestCase):
    def write_rows(self, rows):
        handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
        with handle:
            for item in rows:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_manual_approval_is_accepted(self):
        docs, rejected = load_corpus(self.write_rows([row("1"), row("2")]), False)
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0]["trust"], "manually_verified")
        self.assertEqual(rejected["invalid_metadata"], 0)

    def test_legacy_auto_review_is_not_accepted_by_default(self):
        docs_path = self.write_rows([
            row("1", note="AUTO_REVIEW: structural triage"),
            row("2", note="AUTO_REVIEW: structural triage"),
        ])
        with self.assertRaises(ValueError):
            load_corpus(docs_path, False)

    def test_unverified_requires_explicit_opt_in_and_is_labeled(self):
        docs_path = self.write_rows([
            row("1", status="pending_review", note=""),
            row("2", status="pending_review", note=""),
        ])
        with self.assertRaises(ValueError):
            load_corpus(docs_path, False)
        docs, _ = load_corpus(docs_path, True)
        self.assertTrue(all(d["trust"] == "unverified_experimental" for d in docs))

    def test_hash_mismatch_is_rejected(self):
        bad = row("1")
        bad["content_hash"] = "0" * 64
        docs, rejected = load_corpus(self.write_rows([bad, row("2")]), False)
        self.assertEqual(len(docs), 1)
        self.assertEqual(rejected["invalid_metadata"], 1)

    def test_split_is_stable_and_keeps_documents_separate(self):
        docs = [{"page_id": str(i)} for i in range(10)]
        train_a, valid_a = split_documents(docs)
        train_b, valid_b = split_documents(list(reversed(docs)))
        self.assertEqual([d["page_id"] for d in train_a], [d["page_id"] for d in train_b])
        self.assertEqual([d["page_id"] for d in valid_a], [d["page_id"] for d in valid_b])
        self.assertFalse(set(d["page_id"] for d in train_a) & set(d["page_id"] for d in valid_a))


if __name__ == "__main__":
    unittest.main()
