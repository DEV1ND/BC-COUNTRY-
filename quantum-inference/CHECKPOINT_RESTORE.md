# Quantum Candidate encrypted checkpoint restore

## Storage design

- Training produces a Candidate-only checkpoint.
- The workflow encrypts `quantum-candidate.pt` with AES-256-GCM before upload.
- Only `quantum-candidate.pt.enc` is attached to a GitHub Release. This repository is public, so the release asset is publicly downloadable ciphertext, never plaintext model weights.
- The AES-256-GCM key is not committed, written to release notes, or uploaded as an asset.
- The release tag is pinned as `quantum-candidate-<GitHub run ID>`; the inference service does not follow a floating `latest` pointer.
- Render downloads the selected encrypted asset at startup, authenticates and decrypts it in memory, then loads it with PyTorch `weights_only=True`.
- The local Render filesystem is not used for durable checkpoint storage.

## Required secret configuration

Configure the same 64-character hexadecimal (32-byte) key value directly in both locations; never paste it into chat or commit it:

1. GitHub repository Actions secret: `QUANTUM_CHECKPOINT_ENCRYPTION_KEY`
2. Render Candidate service secret: `QUANTUM_CHECKPOINT_ENCRYPTION_KEY`

The GitHub workflow also requires `JUCKEX_WIKI_ADMIN_TOKEN` to export the private corpus.

After a successful training workflow run, copy the printed release tag (format `quantum-candidate-<run ID>`) into the Render Candidate service environment variable:

- `QUANTUM_CHECKPOINT_RELEASE_TAG` = exact release tag

The service must remain Candidate-only. Do not add these variables to the Production service.

## Safety behavior

- Missing release tag: the API reports the model as not loaded, preserving the current safe fallback.
- Configured tag with missing key, missing asset, invalid URL, oversized asset, wrong key, or altered ciphertext: startup fails closed. It does not silently claim that the model loaded.
- Only the exact encrypted asset name is accepted from the pinned release.
- Release assets are public ciphertext. Anyone can download them, but AES-256-GCM authentication prevents loading altered data and the key remains separate.
- No R2 bucket, paid Render disk, Production service, Cloudflare Worker, D1, or KV mutation is required.

## Limitations

Render Free has an ephemeral filesystem and no persistent disks. The service fetches the pinned encrypted checkpoint at each cold start. This adds startup network dependency and model-load time. It does not persist the decrypted checkpoint between restarts.

## Verification

The Candidate CI suite tests successful decryption, wrong-key rejection, tamper rejection, release tag pinning, asset URL restrictions, and asset size validation. A successful CI run does not mean a real trained checkpoint exists or has been deployed.
