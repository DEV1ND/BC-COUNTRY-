import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generation_sources import build_knowledge_context, serialize_knowledge_sources


def source(**overrides):
    row = {
        "title": "인공지능",
        "url": "https://ko.wikipedia.org/wiki/인공지능",
        "text": "인공지능은 기계가 지능적인 작업을 수행하도록 하는 기술 분야다. " * 12,
        "license": "CC BY-SA 4.0",
        "attribution": "Wikipedia contributors, 인공지능, https://ko.wikipedia.org/wiki/인공지능; licensed under CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/).",
    }
    row.update(overrides)
    return row


class GenerationSourcesTests(unittest.TestCase):
    def test_valid_wikipedia_source_is_serialized(self):
        self.assertEqual(
            serialize_knowledge_sources([source()]),
            [{
                "title": "인공지능",
                "url": "https://ko.wikipedia.org/wiki/인공지능",
                "license": "CC BY-SA 4.0",
                "attribution": source()["attribution"],
            }],
        )

    def test_invalid_url_license_or_attribution_is_not_cited(self):
        rows = [
            source(url="https://example.com/fake"),
            source(url="https://ko.wikipedia.org.evil.example/wiki/인공지능"),
            source(license="CC BY 3.0"),
            source(attribution="출처 표기만 있고 라이선스 정보 없음"),
            source(title=""),
            source(text=""),
        ]
        self.assertEqual(serialize_knowledge_sources(rows), [])
        self.assertEqual(build_knowledge_context(rows), "")

    def test_attribution_must_include_title_url_and_license_terms(self):
        self.assertEqual(
            serialize_knowledge_sources([
                source(attribution="Wikipedia contributors; source page")
            ]),
            [],
        )

    def test_duplicate_sources_are_deduplicated(self):
        row = source()
        self.assertEqual(len(serialize_knowledge_sources([row, row])), 1)

    def test_non_string_source_fields_are_not_coerced(self):
        for field in ("title", "url", "license", "attribution", "text"):
            with self.subTest(field=field):
                self.assertEqual(serialize_knowledge_sources([
                    source(**{field: 123})
                ]), [])

    def test_source_collection_and_excerpt_limits_are_validated(self):
        with self.assertRaises(TypeError):
            serialize_knowledge_sources("not a document list")
        for value in (True, 0, -1, 2001, "80"):
            with self.subTest(excerpt_chars=value):
                with self.assertRaises(ValueError):
                    build_knowledge_context([source()], excerpt_chars=value)

    def test_context_and_citations_use_the_same_validated_documents(self):
        valid = source()
        invalid = source(title="가짜 문서", url="https://example.com/fake")
        context = build_knowledge_context([valid, invalid], excerpt_chars=80)
        citations = serialize_knowledge_sources([valid, invalid])
        self.assertIn("[인공지능]", context)
        self.assertNotIn("가짜 문서", context)
        self.assertEqual([item["title"] for item in citations], ["인공지능"])
        self.assertIn("CC BY-SA 4.0", citations[0]["license"])
        self.assertTrue(citations[0]["attribution"])


if __name__ == "__main__":
    unittest.main()
