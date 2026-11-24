import unittest

from seed_quality import should_skip_wikipedia_seed


class WikipediaSeedPolicyTests(unittest.TestCase):
    def setUp(self):
        self.healthy = {
            "documents": 40,
            "titles": ["인공지능", "컴퓨터", "대한민국"],
            "schema_complete": True,
            "integrity": {"healthy": True},
        }

    def test_skips_seed_when_existing_index_is_complete_and_healthy(self):
        self.assertTrue(should_skip_wikipedia_seed(self.healthy, required_documents=40))

    def test_does_not_skip_when_document_count_is_too_small(self):
        status = {**self.healthy, "documents": 39}
        self.assertFalse(should_skip_wikipedia_seed(status, required_documents=40))

    def test_does_not_skip_when_integrity_is_unhealthy(self):
        status = {**self.healthy, "integrity": {"healthy": False}}
        self.assertFalse(should_skip_wikipedia_seed(status, required_documents=40))

    def test_does_not_skip_when_schema_is_incomplete(self):
        status = {**self.healthy, "schema_complete": False}
        self.assertFalse(should_skip_wikipedia_seed(status, required_documents=40))

    def test_does_not_skip_when_expected_topic_is_missing(self):
        self.assertFalse(should_skip_wikipedia_seed(
            self.healthy, required_documents=40, required_titles=["인공지능", "과학"]
        ))

    def test_skips_only_when_all_expected_topics_exist(self):
        self.assertTrue(should_skip_wikipedia_seed(
            self.healthy, required_documents=40, required_titles=["인공지능", "컴퓨터"]
        ))

    def test_force_reseed_overrides_healthy_cache(self):
        self.assertFalse(should_skip_wikipedia_seed(self.healthy, required_documents=40, force=True))

    def test_invalid_status_fails_closed(self):
        self.assertFalse(should_skip_wikipedia_seed(None, required_documents=40))
        self.assertFalse(should_skip_wikipedia_seed({"documents": "bad"}, required_documents=40))


if __name__ == "__main__":
    unittest.main()
