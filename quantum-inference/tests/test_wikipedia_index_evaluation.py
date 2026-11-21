import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_wikipedia_index import evaluate_connection


class WikipediaIndexEvaluationTests(unittest.TestCase):
    def test_evaluation_is_read_only_and_reports_source_metadata(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""CREATE TABLE knowledge_docs (
            page_id TEXT PRIMARY KEY, title TEXT NOT NULL, url TEXT NOT NULL,
            text TEXT NOT NULL, license TEXT NOT NULL, attribution TEXT NOT NULL)""")
        conn.execute("""CREATE VIRTUAL TABLE knowledge_fts USING fts5(
            page_id UNINDEXED,title,text,tokenize='unicode61')""")
        body = "인공지능은 컴퓨터 과학의 한 분야이며 지능적인 작업을 수행하는 시스템을 연구한다. " * 8
        conn.execute("INSERT INTO knowledge_docs VALUES(?,?,?,?,?,?)",
                     ("1", "인공지능", "https://ko.wikipedia.org/wiki/인공지능",
                      body, "CC BY-SA 4.0", "Wikipedia contributors; title and URL retained"))
        conn.execute("INSERT INTO knowledge_fts VALUES(?,?,?)", ("1", "인공지능", body))
        conn.commit()
        before = conn.execute("SELECT COUNT(*) FROM knowledge_docs").fetchone()[0]
        report = evaluate_connection(conn)
        after = conn.execute("SELECT COUNT(*) FROM knowledge_docs").fetchone()[0]
        self.assertEqual(before, after)
        self.assertEqual(report["database_document_count"], 1)
        self.assertTrue(report["read_only"])
        self.assertFalse(report["model_training"])
        self.assertEqual(report["metadata_audit"]["invalid_license"], 0)
        self.assertEqual(report["metadata_audit"]["missing_attribution"], 0)
        conn.close()


if __name__ == "__main__":
    unittest.main()
