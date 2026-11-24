"""SQLite integration checks for the isolated Candidate D1 nonce migration."""
import sqlite3
import unittest
from pathlib import Path

MIGRATION = Path(__file__).resolve().parent / "cloudflare-workers" / "migrations" / "0002_review_attestation_nonces.sql"
INSERT = (
    "INSERT INTO review_attestation_nonces "
    "(nonce, review_payload_sha256, reviewer_email, reviewer_subject, issued_at, expires_at) "
    "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(nonce) DO NOTHING"
)
ROW = (
    "0123456789abcdef", "a" * 64, "reviewer@example.org", "subject-123",
    "2026-11-25T00:00:00Z", "2026-11-25T00:05:00Z",
)


class ReviewNonceMigrationTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.db.executescript(MIGRATION.read_text(encoding="utf-8"))

    def tearDown(self):
        self.db.close()

    def test_migration_creates_nonce_table_and_expiry_index(self):
        table = self.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='review_attestation_nonces'"
        ).fetchone()
        index = self.db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_review_attestation_nonces_expiry'"
        ).fetchone()
        self.assertIsNotNone(table)
        self.assertIsNotNone(index)

    def test_first_nonce_insert_succeeds_and_duplicate_is_noop(self):
        first = self.db.execute(INSERT, ROW)
        second = self.db.execute(INSERT, ROW)
        self.assertEqual(first.rowcount, 1)
        self.assertEqual(second.rowcount, 0)
        self.assertEqual(self.db.execute(
            "SELECT COUNT(*) FROM review_attestation_nonces WHERE nonce=?", (ROW[0],)
        ).fetchone()[0], 1)

    def test_distinct_nonces_can_be_recorded(self):
        self.db.execute(INSERT, ROW)
        second = (ROW[0] + "_other",) + ROW[1:]
        self.db.execute(INSERT, second)
        self.assertEqual(self.db.execute(
            "SELECT COUNT(*) FROM review_attestation_nonces"
        ).fetchone()[0], 2)

    def test_nonce_length_constraint_rejects_short_values(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute(INSERT, ("short",) + ROW[1:])

    def test_digest_length_constraint_rejects_invalid_values(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute(INSERT, (ROW[0], "bad",) + ROW[2:])

    def test_expiry_cleanup_removes_only_older_records(self):
        self.db.execute(INSERT, ROW)
        self.db.execute(INSERT, (ROW[0] + "_old",) + ROW[1:5] + ("2026-11-24T00:00:00Z",))
        self.db.execute(
            "DELETE FROM review_attestation_nonces WHERE expires_at < ?",
            ("2026-11-25T00:00:00Z",),
        )
        remaining = self.db.execute(
            "SELECT nonce FROM review_attestation_nonces"
        ).fetchall()
        self.assertEqual(remaining, [(ROW[0],)])


if __name__ == "__main__":
    unittest.main()
