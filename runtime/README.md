# Quantum Answer Engine — isolated integration module

## What this adds

runtime/quantum-answer-engine.mjs is a runtime-neutral orchestration module for the JUCKEX-owned Quantum inference service. It performs:

1. Conversation-history normalization and size limits.
2. Retrieval through an injected JUCKEX knowledge-search adapter.
3. Lightweight relevance ranking and filtering.
4. Evidence packaging with untrusted-source boundaries.
5. Answer generation through an injected Quantum-only inference adapter.
6. Server-authored citation metadata for documents actually supplied to Quantum.

It does not call Cloudflare Workers AI, OpenAI, Gemini, Claude, or any other external LLM. It does not train or load model weights itself.

## Integration contract

The host application must provide two functions:

- retrieveDocuments(query, context): return records with title, url, text (or content / snippet), and optional publisher, publishedAt, license, attribution, and revisionId.
- generateWithQuantum({ messages, maxTokens }): call the private JUCKEX Quantum inference service and return either answer text or { text, model, usage }.

Example wiring (illustrative adapter names; connect only after confirming the real service contract):

```js
import { createQuantumAnswerEngine } from "./runtime/quantum-answer-engine.mjs";

const answerQuantum = createQuantumAnswerEngine({
  retrieveDocuments: async (query, context) => knowledgeStore.search(query, context),
  generateWithQuantum: async ({ messages, maxTokens }) => {
    // Implement this adapter against the verified private Quantum inference API.
    // Never replace it with a third-party LLM provider.
    return quantumInference.generate({ messages, maxTokens });
  },
});

const result = await answerQuantum({ question, history });
return Response.json(result);
```

The example deliberately uses abstract adapters: the current production Worker source and Render inference contract have not yet been reconciled, so no endpoint, token binding, payload schema, or deployment target is assumed.

## Citation behavior

- A source must have a valid HTTP(S) URL and non-empty evidence text.
- Low-relevance and invalid sources are omitted.
- Only the selected sources are sent to Quantum and returned in sources.
- Quantum may cite only the exact markers ([S1], [S2], ...) included in its system prompt.
- The UI should render sources as clickable links and should not parse or trust model-generated URLs.
- The host must ensure each source's license and attribution metadata is accurate.

This first version uses lexical relevance ranking. It is a baseline, not semantic search. Replace or combine it with the production Korean-aware retrieval index only after its schema and permissions are verified.

## Tests

Run with Node.js 20 or newer:

```bash
node --test runtime/quantum-answer-engine.test.mjs
```

## Not yet done

- No production Worker was changed.
- No Cloudflare or Render resource was modified.
- No Quantum model was trained, loaded, or deployed.
- The real Render inference adapter and knowledge-store adapter are not wired yet.
- The currently inspected juckex Worker still calls env.AI.run("@cf/google/gemma-4-26b-a4b-it", ...); this module does not silently alter that behavior.
- Do not merge or deploy this module until the actual Quantum endpoint, authentication binding, request/response schema, and test environment are verified.
