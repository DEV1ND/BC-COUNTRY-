# Quantum Lab Worker (isolated source snapshot)

This folder contains a versioned, isolated copy of the deployed `juckex-quantum-lab` Worker, updated to forward Quantum authentication server-side and display Wikipedia citations returned by the inference API.

## Required Cloudflare configuration

Before deploying this source to a separate test Worker, configure:

- Secret `QUANTUM_API_TOKEN`: same value as the inference service's `QUANTUM_API_TOKEN`. Never put this in browser code or commit it.
- Variable `QUANTUM_INFERENCE_URL`: `https://juckex-quantum-inference.onrender.com` (or a separate candidate service URL).

The Worker deliberately refuses live generation if the secret is absent. The token is attached only to the server-to-server `/v1/generate` request. Health checks remain unauthenticated.

## Citation behavior

Only URLs beginning with `https://ko.wikipedia.org/wiki/` are rendered as links. Title, URL, license, and attribution are taken from `knowledge_sources` returned by Quantum. No citation is fabricated when the inference response has no source metadata.

## Safety and deployment boundary

This is a source snapshot in the `quantum-data-repair` branch. It has **not** been deployed to the existing Worker. Validate it in a separate preview/test Worker with test-only credentials before considering any change to the current Lab. It does not change model weights, training data, D1 records, or the production JUCKEX Worker.
