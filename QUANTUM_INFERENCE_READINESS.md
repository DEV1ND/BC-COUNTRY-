# Quantum inference readiness and startup safety

## Why this guard exists

A remote model restore can fail because of network/authentication problems or an incompatible checkpoint. Such a failure is not evidence that the remote model is absent. The API must not treat a freshly initialized Transformer as a usable Quantum model.

## Readiness rules

Inference is ready only when one of these conditions holds:

- The remote Production model was restored successfully.
- Remote storage is unavailable or explicitly reports no model, and a local checkpoint was successfully loaded and validated.

Inference is **not** ready when:

- Remote restore raises an exception, even if a stale local checkpoint exists.
- No remote model exists and no valid local checkpoint was loaded.
- The process has only randomly initialized model parameters.

When not ready:

- `GET /health` returns HTTP 503 and `ok=false`.
- `GET /` reports `status=degraded`.
- `POST /generate` returns HTTP 503 without invoking the model.
- A missing remote model is not bootstrapped from random parameters.
- The service records a machine-readable `inference_block_reason`.

## Test scope

Unit tests cover successful remote restore, validated local fallback, missing weights, remote restore exceptions, and route-level fail-closed checks.

These are code-level checks. They do not replace a real Render startup test against Cloudflare Worker storage. No live service was changed or deployed by this Candidate PR.
