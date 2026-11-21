# Quantum Wikipedia sync reconciliation

This directory contains the source mirror for the deployed `juckex-quantum-wikipedia-explorer` Worker, including its `auto_review.js` module dependency.

## Why this change exists

The old sync routine advanced through approved D1 pages using the inference cache's document count as a SQL OFFSET. That is unsafe: the old cache may contain a different subset/order, so repeated batches can re-import already cached pages and leave other approved pages unsynced.

## New reconciliation protocol

1. The inference API exposes authenticated `GET /admin/knowledge/ids`, returning the exact page IDs currently in its local knowledge cache.
2. The Worker compares those IDs with approved page IDs in D1.
3. It imports the first five missing IDs (200–50,000 characters each, matching the inference API limit) and records the successful sync timestamp.
4. If the Render instance loses its local cache, its ID list becomes empty and the Worker repopulates it from D1 without relying on stale sync timestamps.

D1 remains the canonical knowledge source. Render's SQLite database is treated only as a rebuildable retrieval cache.

## Safe rollout order

1. Deploy the matching inference API code so `/admin/knowledge/ids` exists.
2. Confirm the authenticated endpoint returns `page_ids` and a document count.
3. Deploy this Worker source to the existing isolated Wikipedia explorer Worker.
4. Watch the next scheduled cycles and verify the cache count rises toward the D1 approved count.
5. Run retrieval smoke tests for representative Korean topics and verify returned source URLs, titles, and CC BY-SA 4.0 attribution.

Do not deploy this branch automatically. The Render service currently has a build command that trains a candidate checkpoint; changing that command requires a separate checkpoint-preservation plan and explicit approval. No model training or production promotion is part of this sync fix.

## Licensing

Every imported page must retain its title, source URL, CC BY-SA 4.0 license, and attribution metadata. The inference API currently accepts 200–50,000 characters per page; longer pages must be curated or chunked in a separate, traceable change rather than silently truncated. Retrieval is not model-weight training.
