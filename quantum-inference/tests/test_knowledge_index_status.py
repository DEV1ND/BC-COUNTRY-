import sqlite3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knowledge_index_status import build_index_status


class KnowledgeIndexStatusTests(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("CREATE TABLE knowledge_docs (page_id TEXT PRIMARY KEY)")
        self.conn.executemany(
            "INSERT INTO knowledge_docs(page_id) VALUES(?)",
            [("page-c",), ("page-a",), ("page-b",)],
        )

    def tearDown(self):
        self.conn.close()

    def test_returns_stable_complete_id_manifest(self):
        status = build_index_status(self.conn)
        self.assertEqual(status["documents"], 3)
        self.assertEqual(status["page_ids"], ["page-a", "page-b", "page-c"])
        self.assertTrue(status["page_ids_complete"])

    def test_marks_truncated_manifest_incomplete(self):
        status = build_index_status(self.conn, page_id_limit=2)
        self.assertEqual(status["documents"], 3)
        self.assertEqual(status["page_ids"], ["page-a", "page-b"])
        self.assertFalse(status["page_ids_complete"])

    def test_is_read_only(self):
        before = self.conn.execute("SELECT COUNT(*) FROM knowledge_docs").fetchone()[0]
        build_index_status(self.conn)
        after = self.conn.execute("SELECT COUNT(*) FROM knowledge_docs").fetchone()[0]
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
