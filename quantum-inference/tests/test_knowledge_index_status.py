import hashlib, sqlite3, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knowledge_index_status import build_index_status
from knowledge_provenance import WIKIPEDIA_LICENSE, WIKIPEDIA_LICENSE_URL

class KnowledgeIndexStatusTests(unittest.TestCase):
    def setUp(self):
        self.conn=sqlite3.connect(":memory:")
        self.conn.execute("""CREATE TABLE knowledge_docs(page_id TEXT PRIMARY KEY,title TEXT,url TEXT,text TEXT,
            license TEXT,attribution TEXT,revision_id TEXT,content_hash TEXT)""")
        self.conn.execute("CREATE VIRTUAL TABLE knowledge_fts USING fts5(page_id UNINDEXED,title,text)")
        self.body="인공지능은 컴퓨터 과학의 한 분야이며 지능적 작업을 연구하는 학문이다."
        self.url="https://ko.wikipedia.org/wiki/인공지능"
        self.attr=f"Wikipedia contributors, 인공지능, {self.url}; {WIKIPEDIA_LICENSE} ({WIKIPEDIA_LICENSE_URL})"
        for pid in ("page-c","page-a","page-b"): self.insert_doc(pid)
    def insert_doc(self,pid,body=None,attr=None,indexed=True):
        body=body or self.body
        self.conn.execute("INSERT INTO knowledge_docs VALUES(?,?,?,?,?,?,?,?)",
            (pid,"인공지능",self.url,body,WIKIPEDIA_LICENSE,attr or self.attr,"12345",hashlib.sha256(body.encode()).hexdigest()))
        if indexed: self.conn.execute("INSERT INTO knowledge_fts VALUES(?,?,?)",(pid,"인공지능",body))
    def tearDown(self): self.conn.close()
    def test_manifest_and_integrity(self):
        s=build_index_status(self.conn)
        self.assertEqual(s["page_ids"],["page-a","page-b","page-c"])
        self.assertTrue(s["page_ids_complete"])
        self.assertTrue(s["integrity"]["healthy"])
    def test_manifest_truncation(self):
        s=build_index_status(self.conn,2)
        self.assertFalse(s["page_ids_complete"])
        self.assertEqual(s["page_ids"],["page-a","page-b"])
    def test_detects_hash_provenance_and_fts_drift(self):
        self.insert_doc("page-d",body="변경된 본문",attr="잘못된 출처",indexed=False)
        self.conn.execute("UPDATE knowledge_docs SET content_hash=? WHERE page_id='page-d'",("0"*64,))
        s=build_index_status(self.conn)
        self.assertEqual(s["integrity"]["content_hash_mismatch"],1)
        self.assertEqual(s["integrity"]["invalid_provenance"],1)
        self.assertEqual(s["integrity"]["documents_missing_from_fts"],1)
        self.assertFalse(s["integrity"]["healthy"])
    def test_missing_fts_is_reported(self):
        self.conn.execute("DROP TABLE knowledge_fts")
        s=build_index_status(self.conn)
        self.assertFalse(s["integrity"]["fts_available"])
        self.assertFalse(s["integrity"]["healthy"])
    def test_legacy_schema_is_reported(self):
        conn=sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE TABLE knowledge_docs(page_id TEXT PRIMARY KEY)")
            conn.execute("INSERT INTO knowledge_docs VALUES('legacy')")
            s=build_index_status(conn)
            self.assertFalse(s["schema_complete"])
            self.assertFalse(s["integrity"]["checked"])
        finally: conn.close()

if __name__=="__main__": unittest.main()
