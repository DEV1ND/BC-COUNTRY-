import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from export_wikipedia_corpus import export_corpus


def document(**overrides):
    text = "한국어 위키백과 문서 내용입니다. " * 20
    row = {
        "page_id": "12345",
        "title": "테스트 문서",
        "url": "https://ko.wikipedia.org/wiki/테스트_문서",
        "text": text,
        "license": "CC BY-SA 4.0",
        "attribution": (
            "Wikipedia contributors, 테스트 문서, "
            "https://ko.wikipedia.org/wiki/테스트_문서; licensed under "
            "CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)."
        ),
        "revision_id": "67890",
        "content_hash": hashlib.sha256(text.strip().encode("utf-8")).hexdigest(),
        "status": "pending_review",
        "reviewer_note": "",
        "reviewed_at": None,
        "provenance_review": None,
    }
    row.update(overrides)
    return row


class ExportWikipediaCorpusTests(unittest.TestCase):
    def export_with_pages(self, pages, corpus="unverified", destination=None):
        if destination is None:
            temp_dir = tempfile.TemporaryDirectory()
            self.addCleanup(temp_dir.cleanup)
            destination = Path(temp_dir.name) / "corpus.jsonl"
        with patch("export_wikipedia_corpus.fetch_page", side_effect=pages):
            count = export_corpus("https://example.workers.dev", "token", corpus, destination)
        return count, Path(destination)


    def test_export_rejects_boolean_page_limit_before_network_access(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "corpus.jsonl"
            with patch("export_wikipedia_corpus.fetch_page") as fetch:
                with self.assertRaisesRegex(ValueError, "limit must be an integer"):
                    export_corpus("https://example.workers.dev", "token", "unverified", destination, limit=True)
            fetch.assert_not_called()
            self.assertFalse(destination.exists())

    def test_export_rejects_body_larger_than_import_contract(self):
        text = "가" * 50001
        bad = document(text=text, content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest())
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "corpus.jsonl"
            with patch("export_wikipedia_corpus.fetch_page", return_value={
                "documents": [bad], "has_more": False, "next_after": None
            }):
                with self.assertRaisesRegex(RuntimeError, "outside the 200-50000"):
                    export_corpus("https://example.workers.dev", "token", "unverified", destination)
            self.assertFalse(destination.exists())

    def test_valid_document_is_exported_as_jsonl(self):
        count, path = self.export_with_pages([
            {"documents": [document()], "has_more": False, "next_after": None}
        ])
        self.assertEqual(count, 1)
        row = json.loads(path.read_text(encoding="utf-8").strip())
        self.assertEqual(row["page_id"], "12345")
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_numeric_page_id_is_rejected_instead_of_coerced(self):
        bad = document(page_id=12345)
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "corpus.jsonl"
            with patch("export_wikipedia_corpus.fetch_page", return_value={
                "documents": [bad], "has_more": False, "next_after": None
            }):
                with self.assertRaisesRegex(RuntimeError, "non-string metadata"):
                    export_corpus("https://example.workers.dev", "token", "unverified", destination)
            self.assertFalse(destination.exists())

    def test_hash_mismatch_is_rejected_and_existing_destination_is_preserved(self):
        bad = document(content_hash="0" * 64)
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "corpus.jsonl"
            destination.write_text("previous-safe-file", encoding="utf-8")
            with patch("export_wikipedia_corpus.fetch_page", return_value={
                "documents": [bad], "has_more": False, "next_after": None
            }):
                with self.assertRaisesRegex(RuntimeError, "does not match"):
                    export_corpus("https://example.workers.dev", "token", "unverified", destination)
            self.assertEqual(destination.read_text(encoding="utf-8"), "previous-safe-file")

    def test_invalid_wikipedia_host_is_rejected(self):
        bad = document(url="https://wikipedia.org/wiki/테스트")
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "corpus.jsonl"
            with patch("export_wikipedia_corpus.fetch_page", return_value={
                "documents": [bad], "has_more": False, "next_after": None
            }):
                with self.assertRaisesRegex(RuntimeError, "invalid Korean Wikipedia URL"):
                    export_corpus("https://example.workers.dev", "token", "unverified", destination)
            self.assertFalse(destination.exists())

    def test_manual_corpus_requires_manual_approval_marker(self):
        bad = document(status="approved", reviewer_note="AUTO_REVIEW: passed", reviewed_at="2026-11-23T10:00:00Z", provenance_review={
            "approved": True, "reviewer": "reviewer@example.org",
            "reviewer_subject": "cloudflare-subject-001",
            "identity_provider": "cloudflare_access",
            "reviewer_identity_verified": True,
            "review_note": "독립 자료와 핵심 사실을 대조하여 검토했습니다.",
            "reference_url": "https://example.org/verified-reference",
            "reviewed_at": "2026-11-23T10:00:00.000Z",
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "corpus.jsonl"
            with patch("export_wikipedia_corpus.fetch_page", return_value={
                "documents": [bad], "has_more": False, "next_after": None
            }):
                with self.assertRaisesRegex(RuntimeError, "without human approval"):
                    export_corpus("https://example.workers.dev", "token", "manual", destination)
            self.assertFalse(destination.exists())


    def test_incomplete_attribution_is_rejected(self):
        bad = document(attribution="Wikipedia contributors, 테스트 문서")
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "corpus.jsonl"
            with patch("export_wikipedia_corpus.fetch_page", return_value={
                "documents": [bad], "has_more": False, "next_after": None
            }):
                with self.assertRaisesRegex(RuntimeError, "incomplete or invalid CC BY-SA attribution"):
                    export_corpus("https://example.workers.dev", "token", "unverified", destination)
            self.assertFalse(destination.exists())


    def test_manual_corpus_requires_timezone_aware_review_timestamp(self):
        bad = document(
            status="approved",
            reviewer_note="MANUAL_APPROVAL_V2: independent source checked",
            reviewed_at="2026-11-23 10:00:00",
            provenance_review={
            "approved": True, "reviewer": "reviewer@example.org",
            "reviewer_subject": "cloudflare-subject-001",
            "identity_provider": "cloudflare_access",
            "reviewer_identity_verified": True,
            "review_note": "독립 자료와 핵심 사실을 대조하여 검토했습니다.",
            "reference_url": "https://example.org/verified-reference",
            "reviewed_at": "2026-11-23 10:00:00",
        },
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "corpus.jsonl"
            with patch("export_wikipedia_corpus.fetch_page", return_value={
                "documents": [bad], "has_more": False, "next_after": None
            }):
                with self.assertRaisesRegex(RuntimeError, "must include a timezone"):
                    export_corpus("https://example.workers.dev", "token", "manual", destination)
            self.assertFalse(destination.exists())

    def test_valid_manual_corpus_preserves_review_timestamp(self):
        row = document(
            status="approved",
            reviewer_note="MANUAL_APPROVAL_V2: independent source checked",
            reviewed_at="2026-11-23T10:00:00.000Z",
            provenance_review={
            "approved": True, "reviewer": "reviewer@example.org",
            "reviewer_subject": "cloudflare-subject-001",
            "identity_provider": "cloudflare_access",
            "reviewer_identity_verified": True,
            "reviewed_at": "2026-11-23T10:00:00.000Z",
        },
        )
        count, path = self.export_with_pages([
            {"documents": [row], "has_more": False, "next_after": None}
        ], corpus="manual")
        exported = json.loads(path.read_text(encoding="utf-8").strip())
        self.assertEqual(count, 1)
        self.assertEqual(exported["reviewed_at"], "2026-11-23T10:00:00.000Z")
        self.assertTrue(exported["provenance_review"]["reviewer_identity_verified"])


    def test_manual_corpus_rejects_legacy_approval_without_verified_identity(self):
        row = document(
            status="approved",
            reviewer_note="MANUAL_APPROVAL_V2: independent source checked",
            reviewed_at="2026-11-23T10:00:00.000Z",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "corpus.jsonl"
            with patch("export_wikipedia_corpus.fetch_page", return_value={
                "documents": [row], "has_more": False, "next_after": None
            }):
                with self.assertRaisesRegex(RuntimeError, "structured human review record"):
                    export_corpus("https://example.workers.dev", "token", "manual", destination)
            self.assertFalse(destination.exists())

    def test_duplicate_page_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "corpus.jsonl"
            with patch("export_wikipedia_corpus.fetch_page", return_value={
                "documents": [document(), document()], "has_more": False, "next_after": None
            }):
                with self.assertRaisesRegex(RuntimeError, "duplicate page ID"):
                    export_corpus("https://example.workers.dev", "token", "unverified", destination)
            self.assertFalse(destination.exists())

    def test_pagination_cursor_must_be_a_string_and_advance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "corpus.jsonl"
            with patch("export_wikipedia_corpus.fetch_page", return_value={
                "documents": [], "has_more": True, "next_after": 123
            }):
                with self.assertRaisesRegex(RuntimeError, "cursor did not advance"):
                    export_corpus("https://example.workers.dev", "token", "unverified", destination)
            self.assertFalse(destination.exists())

    def test_non_boolean_has_more_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "corpus.jsonl"
            with patch("export_wikipedia_corpus.fetch_page", return_value={
                "documents": [], "has_more": "false", "next_after": None
            }):
                with self.assertRaisesRegex(RuntimeError, "must be boolean"):
                    export_corpus("https://example.workers.dev", "token", "unverified", destination)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
