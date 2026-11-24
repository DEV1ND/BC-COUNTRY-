import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path

from bpe_tokenizer import BPETokenizer
from wikipedia_pretrain import batch_windows, load_corpus, make_windows, split_documents


def row(page_id, status="approved", note="MANUAL_APPROVAL_V2: reviewed against source"):
    text = ("한국어 문서 내용입니다. " * 20).strip()
    return {
        "page_id": page_id, "title": "테스트 문서",
        "url": "https://ko.wikipedia.org/wiki/테스트",
        "revision_id": "12345",
        "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text": text, "license": "CC BY-SA 4.0",
        "attribution": "위키백과 contributors", "status": status,
        "reviewer_note": note,
        "reviewed_at": "2026-11-24T12:00:00.000Z" if status == "approved" else None,
        "provenance_review": ({
            "approved": True, "reviewer": "reviewer@example.org",
            "reviewer_subject": "cloudflare-subject-001",
            "identity_provider": "cloudflare_access",
            "reviewer_identity_verified": True,
            "reviewed_at": "2026-11-24T12:00:00.000Z",
        } if status == "approved" and note.startswith("MANUAL_APPROVAL_V2:") else None),
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
        self.assertEqual(docs[0]["trust"], "identity_verified_human_review")
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


    def test_legacy_manual_note_without_authenticated_identity_is_rejected(self):
        legacy = row("legacy", note="MANUAL_APPROVAL_V2: reviewed against source")
        legacy.pop("provenance_review")
        legacy.pop("reviewed_at")
        path = self.write_rows([
            legacy,
            row("valid-2", status="pending_review", note=""),
            row("valid-3", status="pending_review", note=""),
        ])
        docs, rejected = load_corpus(path, True)
        self.assertEqual(len(docs), 2)
        self.assertEqual(rejected["invalid_metadata"], 1)
        self.assertTrue(all(doc["trust"] == "unverified_experimental" for doc in docs))

    def test_approved_row_requires_cloudflare_access_identity_and_timestamp(self):
        for field, value in (
            ("reviewer_identity_verified", False),
            ("identity_provider", "shared_admin_token"),
            ("reviewer_subject", ""),
        ):
            with self.subTest(field=field):
                bad = row("bad-" + field)
                bad["provenance_review"][field] = value
                docs, rejected = load_corpus(
                    self.write_rows([bad, row("good-1"), row("good-2")]), False
                )
                self.assertEqual(len(docs), 2)
                self.assertEqual(rejected["invalid_metadata"], 1)
        bad_time = row("bad-time")
        bad_time["reviewed_at"] = "2026-11-24T12:00:00"
        bad_time["provenance_review"]["reviewed_at"] = bad_time["reviewed_at"]
        docs, rejected = load_corpus(
            self.write_rows([bad_time, row("good-1"), row("good-2")]), False
        )
        self.assertEqual(len(docs), 2)
        self.assertEqual(rejected["invalid_metadata"], 1)

    def test_approved_identity_metadata_cannot_be_downgraded_with_unverified_flag(self):
        bad = row("legacy")
        bad["provenance_review"] = None
        bad_two = row("legacy-two")
        bad_two["provenance_review"] = None
        with self.assertRaises(ValueError):
            load_corpus(self.write_rows([bad, bad_two]), True)

    def test_unsupported_status_is_rejected(self):
        bad = row("bad-status", status="manually_approved")
        docs, rejected = load_corpus(self.write_rows([bad, row("good-1"), row("good-2")]), True)
        self.assertEqual(len(docs), 2)
        self.assertEqual(rejected["unsupported_status"], 1)

    def test_hash_mismatch_is_rejected(self):
        bad = row("1")
        bad["content_hash"] = "0" * 64
        docs, rejected = load_corpus(self.write_rows([bad, row("2"), row("3")]), False)
        self.assertEqual(len(docs), 2)
        self.assertEqual(rejected["invalid_metadata"], 1)

    def test_provenance_fields_are_not_coerced_from_non_strings(self):
        for field, value in (
            ("page_id", 123),
            ("title", True),
            ("url", 456),
            ("revision_id", 12345),
            ("content_hash", 123),
            ("attribution", ["contributors"]),
            ("text", {"body": "문서"}),
            ("status", True),
            ("reviewer_note", 99),
        ):
            with self.subTest(field=field):
                bad = row("bad-" + field)
                bad[field] = value
                docs, rejected = load_corpus(
                    self.write_rows([bad, row("good-a"), row("good-b")]), False
                )
                self.assertEqual(len(docs), 2)
                self.assertEqual(rejected["invalid_metadata"], 1)

    def test_license_must_match_exactly_without_type_coercion(self):
        for license_value in (None, 123, ["CC BY-SA 4.0"], "cc by-sa 4.0"):
            with self.subTest(license_value=license_value):
                bad = row("bad-license")
                bad["license"] = license_value
                docs, rejected = load_corpus(
                    self.write_rows([bad, row("good-a"), row("good-b")]), False
                )
                self.assertEqual(len(docs), 2)
                self.assertEqual(rejected["invalid_metadata"], 1)


    def test_validation_fraction_and_minimum_documents_are_checked(self):
        docs = [{"page_id": "a"}, {"page_id": "b"}]
        for value in (0, 1, -0.1, 1.1, float("nan"), float("inf"), "bad"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    split_documents(docs, validation_fraction=value)
        with self.assertRaises(ValueError):
            split_documents([{"page_id": "only-one"}])
        with self.assertRaises(ValueError):
            split_documents([{"page_id": ""}, {"page_id": "b"}])
        with self.assertRaisesRegex(ValueError, "page_id"):
            split_documents([{"page_id": 123}, {"page_id": "b"}])

    def test_revisions_of_one_page_never_cross_train_validation_boundary(self):
        docs = [
            {"page_id": "same-page", "revision_id": "1"},
            {"page_id": "same-page", "revision_id": "2"},
            {"page_id": "other-page", "revision_id": "1"},
            {"page_id": "third-page", "revision_id": "1"},
        ]
        train, valid = split_documents(docs, validation_fraction=0.34)
        train_pages = {doc["page_id"] for doc in train}
        valid_pages = {doc["page_id"] for doc in valid}
        self.assertFalse(train_pages & valid_pages)
        same_page_revisions = [
            doc for doc in train + valid if doc["page_id"] == "same-page"
        ]
        self.assertEqual(len(same_page_revisions), 2)
        self.assertTrue(
            all(doc["page_id"] in train_pages for doc in same_page_revisions)
            or all(doc["page_id"] in valid_pages for doc in same_page_revisions)
        )

    def test_split_is_stable_and_keeps_documents_separate(self):
        docs = [{"page_id": str(i)} for i in range(10)]
        train_a, valid_a = split_documents(docs)
        train_b, valid_b = split_documents(list(reversed(docs)))
        self.assertEqual({d["page_id"] for d in train_a}, {d["page_id"] for d in train_b})
        self.assertEqual({d["page_id"] for d in valid_a}, {d["page_id"] for d in valid_b})
        self.assertFalse(set(d["page_id"] for d in train_a) & set(d["page_id"] for d in valid_a))


class TrainingWindowIntegrityTests(unittest.TestCase):
    def test_long_document_tail_tokens_are_not_dropped(self):
        tokenizer = BPETokenizer(initial_symbols=["가"])
        windows, token_count = make_windows([{"text": "가" * 600}], tokenizer)
        self.assertEqual(token_count, 600)
        self.assertEqual([len(window) for window in windows], [257, 257, 88])
        self.assertEqual(sum(len(window) - 1 for window in windows), token_count - 1)
        self.assertEqual(windows[-1][-1], tokenizer.encode("가")[0])

    def test_windows_never_join_tokens_from_different_documents(self):
        tokenizer = BPETokenizer(initial_symbols=["가", "나"])
        windows, _ = make_windows(
            [{"text": "가" * 300}, {"text": "나" * 300}], tokenizer
        )
        self.assertEqual(len(windows), 4)
        for window in windows:
            self.assertEqual(len(set(window)), 1)

    def test_batch_padding_is_excluded_from_next_token_loss(self):
        random.seed(7)
        x, y = batch_windows([[1, 2, 3]], batch_size=1, device="cpu")
        self.assertEqual(x[0, :2].tolist(), [1, 2])
        self.assertEqual(y[0, :2].tolist(), [2, 3])
        self.assertEqual(x[0, 2:].count_nonzero().item(), 0)
        self.assertTrue((y[0, 2:] == -100).all().item())

    def test_batch_rejects_empty_windows(self):
        with self.assertRaisesRegex(ValueError, "비어 있지 않은"):
            batch_windows([], batch_size=1, device="cpu")

    def test_batch_rejects_invalid_batch_size(self):
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size):
                with self.assertRaisesRegex(ValueError, "batch_size"):
                    batch_windows([[1, 2]], batch_size=size, device="cpu")

    def test_batch_rejects_short_or_malformed_windows(self):
        for windows in ([[]], [[1]], [[1, -2]], [[1, True]], [["1", 2]]):
            with self.subTest(windows=windows):
                with self.assertRaises(ValueError):
                    batch_windows(windows, batch_size=1, device="cpu")

    def test_batch_rejects_windows_longer_than_context(self):
        with self.assertRaisesRegex(ValueError, "BLOCK 크기"):
            batch_windows([list(range(258))], batch_size=1, device="cpu")

    def test_batch_rejects_token_ids_outside_vocabulary(self):
        with self.assertRaisesRegex(ValueError, "vocab_size 범위"):
            batch_windows([[1, 2, 8]], batch_size=1, device="cpu", vocab_size=8)

    def test_batch_rejects_invalid_vocabulary_size(self):
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size):
                with self.assertRaisesRegex(ValueError, "vocab_size"):
                    batch_windows([[1, 2]], batch_size=1, device="cpu", vocab_size=size)


if __name__ == "__main__":
    unittest.main()
