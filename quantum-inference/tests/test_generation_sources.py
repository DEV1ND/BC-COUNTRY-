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
        "attribution": "Wikipedia contributors; title and URL retained",
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
                "attribution": "Wikipedia contributors; title and URL retained",
            }],
        )

    def test_invalid_url_license_or_attribution_is_not_cited(self):
        rows = [
            source(url="https://example.com/fake"),
            source(license="CC BY 3.0"),
            source(attribution=" "),
            source(title=""),
            source(text=""),
        ]
        self.assertEqual(serialize_knowledge_sources(rows), [])
        self.assertEqual(build_knowledge_context(rows), "")

    def test_duplicate_sources_are_deduplicated(self):
        row = source()
        self.assertEqual(len(serialize_knowledge_sources([row, row])), 1)

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
