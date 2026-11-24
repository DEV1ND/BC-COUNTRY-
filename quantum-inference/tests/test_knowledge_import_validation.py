import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knowledge_import_validation import validate_import_document, validate_import_documents


def valid_document(**overrides):
    body = "위키백과 문서 본문 검증용 문장입니다. " * 12
    row = {
        "page_id": "12345",
        "title": "인공지능",
        "url": "https://ko.wikipedia.org/wiki/인공지능",
        "text": body,
        "license": "CC BY-SA 4.0",
        "attribution": (
            "Wikipedia contributors, 인공지능, "
            "https://ko.wikipedia.org/wiki/인공지능; "
            "licensed under CC BY-SA 4.0 "
            "(https://creativecommons.org/licenses/by-sa/4.0/)."
        ),
        "revision_id": "987654",
        "content_hash": hashlib.sha256(body.strip().encode("utf-8")).hexdigest(),
    }
    row.update(overrides)
    return row


class KnowledgeImportValidationTests(unittest.TestCase):
    def test_valid_document_is_returned_with_trimmed_string_fields(self):
        row = valid_document(title=" 인공지능 ")
        result = validate_import_document(row)
        self.assertEqual(result["title"], "인공지능")
        self.assertEqual(result["page_id"], "12345")

    def test_every_required_field_rejects_implicit_type_coercion(self):
        for field in (
            "page_id", "title", "url", "text", "license",
            "attribution", "revision_id", "content_hash",
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "문자열"):
                    validate_import_document(valid_document(**{field: 123}))

    def test_boolean_is_not_accepted_as_a_string_identifier(self):
        with self.assertRaises(ValueError):
            validate_import_document(valid_document(page_id=True))

    def test_wikipedia_provenance_is_mandatory(self):
        with self.assertRaisesRegex(ValueError, "저작자 표시"):
            validate_import_document(valid_document(url="https://example.com/page"))

    def test_revision_id_must_be_digits(self):
        with self.assertRaisesRegex(ValueError, "revision_id"):
            validate_import_document(valid_document(revision_id="rev-123"))

    def test_content_hash_must_be_sha256_and_match_body(self):
        for value in ("abc", "A" * 64, "0" * 64):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_import_document(valid_document(content_hash=value))

    def test_document_text_length_is_bounded(self):
        with self.assertRaisesRegex(ValueError, "200~50000"):
            validate_import_document(valid_document(text="짧은 본문"))

    def test_batch_must_contain_one_to_ten_documents(self):
        for value in (None, [], [valid_document()] * 11, "invalid"):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(ValueError):
                    validate_import_documents(value)

    def test_entire_batch_is_validated_before_caller_can_write(self):
        valid = valid_document()
        invalid = valid_document(revision_id="not-a-number")
        with self.assertRaises(ValueError):
            validate_import_documents([valid, invalid])

    def test_batch_rejects_duplicate_page_ids_before_database_work(self):
        first = valid_document(page_id="same")
        second = valid_document(page_id="same", title="기계 학습",
            url="https://ko.wikipedia.org/wiki/기계_학습",
            attribution=(
                "Wikipedia contributors, 기계 학습, "
                "https://ko.wikipedia.org/wiki/기계_학습; "
                "licensed under CC BY-SA 4.0 "
                "(https://creativecommons.org/licenses/by-sa/4.0/)."
            ))
        with self.assertRaisesRegex(ValueError, "page_id는 중복"):
            validate_import_documents([first, second])

    def test_valid_batch_preserves_order_and_identity(self):
        first = valid_document(page_id="1")
        second = valid_document(page_id="2", title="기계 학습",
            url="https://ko.wikipedia.org/wiki/기계_학습",
            attribution=(
                "Wikipedia contributors, 기계 학습, "
                "https://ko.wikipedia.org/wiki/기계_학습; "
                "licensed under CC BY-SA 4.0 "
                "(https://creativecommons.org/licenses/by-sa/4.0/)."
            ))
        result = validate_import_documents([first, second])
        self.assertEqual([item["page_id"] for item in result], ["1", "2"])


if __name__ == "__main__":
    unittest.main()
