import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knowledge_provenance import WIKIPEDIA_LICENSE, WIKIPEDIA_LICENSE_URL, has_valid_wikipedia_attribution, is_valid_wikipedia_url

class KnowledgeProvenanceTests(unittest.TestCase):
    def test_accepts_canonical_url(self):
        self.assertTrue(is_valid_wikipedia_url("https://ko.wikipedia.org/wiki/인공지능"))

    def test_rejects_lookalike_and_noncanonical_urls(self):
        for url in (
            "http://ko.wikipedia.org/wiki/인공지능",
            "https://ko.wikipedia.org.evil.example/wiki/인공지능",
            "https://evil.example@ko.wikipedia.org.evil.example/wiki/가짜",
            "https://ko.wikipedia.org:443/wiki/인공지능",
            "https://ko.wikipedia.org/wiki/",
            "https://ko.wikipedia.org/wiki/인공지능?oldid=1",
            "https://ko.wikipedia.org/wiki/인공지능#부분",
            "",
        ):
            with self.subTest(url=url):
                self.assertFalse(is_valid_wikipedia_url(url))

    def test_requires_complete_attribution(self):
        url = "https://ko.wikipedia.org/wiki/인공지능"
        attribution = f"Wikipedia contributors, 인공지능, {url}; licensed under {WIKIPEDIA_LICENSE} ({WIKIPEDIA_LICENSE_URL})."
        self.assertTrue(has_valid_wikipedia_attribution("인공지능", url, WIKIPEDIA_LICENSE, attribution))
        self.assertFalse(has_valid_wikipedia_attribution("다른 제목", url, WIKIPEDIA_LICENSE, attribution))
        self.assertFalse(has_valid_wikipedia_attribution("인공지능", url, WIKIPEDIA_LICENSE, "출처만 있음"))
        self.assertFalse(has_valid_wikipedia_attribution("인공지능", url, "CC BY 3.0", attribution))

if __name__ == "__main__":
    unittest.main()
