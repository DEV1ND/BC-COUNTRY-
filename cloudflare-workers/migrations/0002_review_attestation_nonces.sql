-- Candidate-only single-use nonce registry for signed human-review attestations.
-- Apply only to the isolated Quantum Candidate D1 database.
CREATE TABLE IF NOT EXISTS review_attestation_nonces (
  nonce TEXT PRIMARY KEY NOT NULL
    CHECK (length(nonce) BETWEEN 16 AND 128),
  review_payload_sha256 TEXT NOT NULL
    CHECK (length(review_payload_sha256) = 64),
  reviewer_email TEXT NOT NULL,
  reviewer_subject TEXT NOT NULL,
  issued_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_review_attestation_nonces_expiry
  ON review_attestation_nonces(expires_at);
