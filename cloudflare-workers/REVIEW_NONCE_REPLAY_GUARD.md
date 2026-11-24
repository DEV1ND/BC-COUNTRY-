# Candidate review nonce replay protection

The D1 table in `migrations/0002_review_attestation_nonces.sql` is isolated to the
Quantum Candidate database. The nonce is the primary key, so concurrent attempts
to record the same nonce cannot both succeed.

## Required call order

1. Authenticate the service caller.
2. Verify the HMAC signature with `verify_hmac_review_attestation` semantics.
3. Verify the exact review payload digest, reviewer binding, provider and expiry.
4. Only then call `recordVerifiedReviewNonce(db, attestation)`.
5. Accept the review only when it returns `true`. A `false` result means replay or
   duplicate use and must be rejected.

**Never call this helper before cryptographic verification.** It stores nonce metadata
but does not verify the HMAC signature itself. Do not treat the mere presence of a
nonce row as proof that a person was authenticated.

The helper and migration are not connected to a live route or database binding in
this PR. No D1 migration is applied automatically. The isolated Candidate database
must be identified and backed up before any manual migration. Production databases
must never receive this migration.

Expired nonce cleanup is explicit through `pruneExpiredReviewNonces`; schedule it
only after the Candidate deployment and retention policy are approved.
