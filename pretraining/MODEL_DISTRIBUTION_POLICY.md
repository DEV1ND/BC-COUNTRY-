# Quantum hosted-model and distribution policy

Status: proposed implementation policy on an isolated development branch
Date: 2026-11-22

## Product model

JUCKEX Quantum is a proprietary, hosted AI service. Users interact with it through the JUCKEX application and authenticated service APIs. The service does not provide downloadable model weights or public checkpoint files.

This is an architectural choice inspired by the hosted-service model used by commercial AI providers. It is not a claim that JUCKEX has the same legal status, data rights, training corpus, safeguards, or terms as OpenAI or any other provider.

## Distribution boundaries

- Production and candidate model weights remain in controlled private storage.
- Candidate checkpoints are accessible only to authorized maintainers and isolated evaluation jobs.
- The public application exposes inference endpoints, not filesystem paths, object-store URLs, checkpoint downloads, tokenizer training artifacts, or raw training manifests.
- API responses must not reveal secrets, internal prompts, private corpus text, or model checkpoint contents.
- Service access is governed by JUCKEX terms, privacy notices, rate limits, and applicable law.
- No candidate is promoted automatically. Promotion requires recorded evaluation results and explicit maintainer approval.
- A hosted-only architecture reduces redistribution of weights but does not itself resolve copyright, database-right, privacy, or license obligations arising from training or outputs.

## Data policy

### Wikipedia and Wikimedia material

Wikipedia is a desired candidate source for Quantum pretraining, not merely a retrieval-only source. However, Wikimedia content is commonly available under CC BY-SA 4.0 and some content may have additional or different terms.

Before any Wikipedia text is admitted to a training manifest:
1. Preserve the exact dump/version, language, checksum, retrieval date, page and revision identifiers, article/revision URLs, attribution, license and third-party notices.
2. Filter content with incompatible or unclear rights, personal information, spam, duplicates, and unsuitable material.
3. Record modifications and normalization.
4. Obtain a documented rights review for the intended private training and hosted inference use, including whether model weights or outputs could create obligations under the applicable license and law.
5. Keep the candidate checkpoint private and do not distribute weights unless a separate review authorizes that distribution.
6. Keep a source-level removal and audit process for future training runs.

A private hosted model is not automatically exempt from CC BY-SA. Whether trained weights are an adaptation and what obligations apply is unsettled and fact-dependent. This file is an engineering policy, not legal advice.

### Other permitted data

Use public, licensed, contracted, or user-provided data only where the specific source terms permit the intended model training and hosted service. A dataset-level license label alone is insufficient where underlying records have separate rights. Preserve per-record provenance and attribution.

### Additional supervised fine-tuning

Use verified JUCKEX-authored or otherwise authorized training examples. Keep training, validation, and final evaluation datasets separate. Existing production Q&A tables and fixed evaluation questions remain excluded unless a separate documented review explicitly approves a different use.

## Runtime answer-generation policy

Quantum itself must generate the final answer. No external LLM API may be used as the answer generator or silently substituted for Quantum.

The answer engine should:
1. Interpret the user's request and conversation context.
2. Decide whether external/current knowledge is needed.
3. Retrieve relevant, permitted documents when needed.
4. Provide the retrieved evidence and provenance to Quantum.
5. Generate the answer with Quantum's own model.
6. Return citations only for documents actually retrieved and supplied as evidence; citations are server-authored metadata, not model-invented URLs.
7. State uncertainty when evidence is weak or unavailable.

## Model lifecycle

- Production: currently approved live model; never overwritten by training jobs.
- Candidate: isolated training output.
- Evaluation: independent benchmark, safety, memorization, retrieval-grounding, Korean quality, latency, and regression tests.
- Archive: immutable prior checkpoints and evaluation records.

Candidate-to-Production promotion requires explicit approval, a rollback plan, and a recorded artifact hash. Training jobs must never write to Production paths.

## Current implementation gate

This policy does not change the trainer allowlist, approve any Wikipedia page, start training, alter Cloudflare or Render resources, or deploy a model. The current trainer still rejects CC BY-SA 4.0. A separate reviewed change is required after the source-specific rights gate is complete.
