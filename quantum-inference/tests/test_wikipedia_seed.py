import hashlib
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wikipedia_seed import seed_wikipedia, LICENSE, DEFAULT_TITLES


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


def page(page_id, title, text, revision_id="12345", url=None):
    return {
        "pageid": page_id,
        "title": title,
        "fullurl": url or "https://ko.wikipedia.org/wiki/" + title,
        "extract": text,
        "lastrevid": int(revision_id) if revision_id else None,
    }


class WikipediaSeedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp.name) / "knowledge.sqlite3")
        self.body = ("인공지능은 컴퓨터 과학의 한 분야이며 지능적인 작업을 수행하는 시스템을 연구한다. " * 8).strip()
        self.payload = {"query": {"pages": [
            page("1", "인공지능", self.body),
            page("2", "잘못된 출처", self.body, url="https://example.com/not-wikipedia"),
            page("3", "리비전 누락", self.body, revision_id=""),
        ]}}

    def tearDown(self):
        self.temp.cleanup()

    def run_seed(self, payload=None):
        response = FakeResponse(payload or self.payload)
        with patch.dict(os.environ, {"QUANTUM_WIKIPEDIA_SEED_TITLES": "인공지능|잘못된 출처|리비전 누락"}):
            with patch("urllib.request.urlopen", return_value=response):
                return seed_wikipedia(self.db_path)

    def test_imports_only_valid_wikipedia_page_with_full_provenance(self):
        report = self.run_seed()
        self.assertEqual(report["imported_this_start"], 1)
        self.assertEqual(report["skip_reasons"]["invalid_or_missing_url"], 1)
        self.assertEqual(report["skip_reasons"]["missing_revision_id"], 1)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT title,url,text,license,attribution,revision_id,content_hash FROM knowledge_docs"
            ).fetchone()
            self.assertEqual(row[0], "인공지능")
            self.assertTrue(row[1].startswith("https://ko.wikipedia.org/wiki/"))
            self.assertEqual(row[2], self.body)
            self.assertEqual(row[3], LICENSE)
            self.assertIn("Wikipedia contributors", row[4])
            self.assertEqual(row[5], "12345")
            self.assertEqual(row[6], hashlib.sha256(self.body.encode("utf-8")).hexdigest())
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM knowledge_fts").fetchone()[0], 1)

    def test_reimport_updates_same_page_without_duplicate_fts_rows(self):
        self.run_seed()
        changed = self.body + " 추가된 설명 문장입니다."
        changed_payload = {"query": {"pages": [page("1", "인공지능", changed, "12346")]}}
        report = self.run_seed(changed_payload)
        self.assertEqual(report["imported_this_start"], 1)
        with sqlite3.connect(self.db_path) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM knowledge_docs").fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM knowledge_fts").fetchone()[0], 1)
            row = conn.execute("SELECT revision_id,content_hash,text FROM knowledge_docs").fetchone()
            self.assertEqual(row[0], "12346")
            self.assertEqual(row[1], hashlib.sha256(changed.encode("utf-8")).hexdigest())
            self.assertEqual(row[2], changed)

    def test_existing_legacy_schema_is_migrated_in_place(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE knowledge_docs (
                page_id TEXT PRIMARY KEY, title TEXT NOT NULL, url TEXT NOT NULL,
                text TEXT NOT NULL, license TEXT NOT NULL, attribution TEXT NOT NULL,
                imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
        self.run_seed()
        with sqlite3.connect(self.db_path) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(knowledge_docs)")}
            self.assertIn("revision_id", columns)
            self.assertIn("content_hash", columns)

    def test_empty_or_short_extract_is_not_imported(self):
        payload = {"query": {"pages": [page("4", "짧은 문서", "짧은 내용", "12347")]}}
        report = self.run_seed(payload)
        self.assertEqual(report["imported_this_start"], 0)
        self.assertEqual(report["skip_reasons"]["extract_too_short"], 1)


    def test_empty_batch_retries_each_title(self):
        from urllib.parse import parse_qs, urlparse

        calls = []
        titles = ["인공지능", "컴퓨터"]

        def fake_urlopen(request, timeout=30):
            requested = parse_qs(urlparse(request.full_url).query).get("titles", [""])[0]
            calls.append(requested)
            if "|" in requested:
                return FakeResponse({"query": {"pages": []}})
            return FakeResponse({"query": {"pages": [
                page(str(100 + len(calls)), requested, self.body, "12345")
            ]}})

        with patch.dict(os.environ, {"QUANTUM_WIKIPEDIA_SEED_TITLES": "|".join(titles)}):
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                report = seed_wikipedia(self.db_path)

        self.assertEqual(report["fetch_mode"], "per_title_fallback")
        self.assertEqual(report["returned_pages"], 2)
        self.assertEqual(report["imported_this_start"], 2)
        self.assertEqual(report["total_documents"], 2)
        self.assertEqual(calls, ["인공지능|컴퓨터", "인공지능", "컴퓨터"])

    def test_partial_batch_retries_missing_titles_individually(self):
        from urllib.parse import parse_qs, urlparse
        calls = []
        titles = ["인공지능", "두번째 주제"]
        def fake_urlopen(request, timeout=30):
            requested = parse_qs(urlparse(request.full_url).query).get("titles", [""])[0]
            calls.append(requested)
            if "|" in requested:
                return FakeResponse({"query": {"pages": [
                    page("1", "인공지능", self.body, "12345"),
                    {"title": "두번째 주제", "missing": True},
                ]}})
            return FakeResponse({"query": {"pages": [
                page("2", "두번째 주제", self.body + " 추가 문장입니다.", "12346")
            ]}})
        with patch.dict(os.environ, {"QUANTUM_WIKIPEDIA_SEED_TITLES": "|".join(titles)}):
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                report = seed_wikipedia(self.db_path)
        self.assertEqual(report["fetch_mode"], "batch_plus_per_title_fallback")
        self.assertEqual(report["imported_this_start"], 2)
        self.assertEqual(report["fallback_attempted"], 1)
        self.assertEqual(report["fallback_failures"], 0)
        self.assertEqual(calls, ["인공지능|두번째 주제", "두번째 주제"])
        with sqlite3.connect(self.db_path) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM knowledge_docs").fetchone()[0], 2)

    def test_batch_query_omits_revision_limit_incompatible_with_multiple_titles(self):
        from urllib.parse import parse_qs, urlparse

        captured = {}

        def fake_urlopen(request, timeout=30):
            captured.update(parse_qs(urlparse(request.full_url).query))
            return FakeResponse({"query": {"pages": [
                page("1", "인공지능", self.body, "12345"),
                page("2", "컴퓨터", self.body, "12346"),
            ]}})

        with patch.dict(os.environ, {"QUANTUM_WIKIPEDIA_SEED_TITLES": "인공지능|컴퓨터"}):
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                report = seed_wikipedia(self.db_path)

        self.assertEqual(report["fetch_mode"], "batch")
        self.assertEqual(report["imported_this_start"], 2)
        self.assertNotIn("rvlimit", captured)
        self.assertEqual(captured.get("titles"), ["인공지능|컴퓨터"])

    def test_default_seed_covers_40_unique_topics(self):
        self.assertEqual(len(DEFAULT_TITLES), 40)
        self.assertEqual(len({title.casefold() for title in DEFAULT_TITLES}), 40)
        self.assertIn("데이터베이스", DEFAULT_TITLES)
        self.assertIn("인권", DEFAULT_TITLES)

    def test_requested_titles_are_deduplicated_and_capped_at_50(self):
        from urllib.parse import parse_qs, urlparse

        captured = {}
        titles = [f"주제{i}" for i in range(55)]
        titles[1] = titles[0].upper()

        def fake_urlopen(request, timeout=30):
            captured.update(parse_qs(urlparse(request.full_url).query))
            return FakeResponse({"query": {"pages": [
                page("1", "주제0", self.body, "12345")
            ]}})

        with patch.dict(os.environ, {"QUANTUM_WIKIPEDIA_SEED_TITLES": "|".join(titles)}):
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                report = seed_wikipedia(self.db_path)

        requested = captured["titles"][0].split("|")
        self.assertEqual(len(requested), 50)
        self.assertEqual(len(set(title.casefold() for title in requested)), 50)
        self.assertEqual(report["requested"], 50)

    def test_api_error_is_not_reported_as_empty_success(self):
        with patch.dict(os.environ, {"QUANTUM_WIKIPEDIA_SEED_TITLES": "인공지능"}):
            with patch("urllib.request.urlopen",
                       return_value=FakeResponse({"error": {"code": "ratelimited", "info": "slow down"}})):
                with self.assertRaisesRegex(RuntimeError, "ratelimited"):
                    seed_wikipedia(self.db_path)


class WikipediaD1MigrationTests(unittest.TestCase):
    def test_d1_schema_creates_expected_tables_and_constraints(self):
        migration = Path(__file__).resolve().parents[2] / "cloudflare-workers" / "migrations" / "0001_wikipedia_knowledge.sql"
        self.assertTrue(migration.exists())
        with sqlite3.connect(":memory:") as conn:
            conn.executescript(migration.read_text(encoding="utf-8"))
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertTrue({"settings", "search_topics", "pages", "crawl_runs"}.issubset(tables))
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM search_topics").fetchone()[0], 12)
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    """INSERT INTO pages(id,provider,title,url,domain,content_hash,content,status,
                       license_status,license_name,attribution)
                       VALUES('x','wikipedia','bad','https://example.com','example.com',?,'body',
                       'approved','attribution_required','CC BY-SA 4.0','source')""",
                    ("a" * 64,),
                )


if __name__ == "__main__":
    unittest.main()
