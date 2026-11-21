# Quantum pretraining data intake review

Review date: 2026-11-22

## Decisions

| Source | Current decision | Why |
|---|---|---|
| Mozilla Common Voice Scripted Speech 27.0 Korean | Supplemental candidate; final human approval pending | Bundled README states CC0. The archive is small and consists of ASR prompt sentences, not broad knowledge text. The original retrieval date is not recorded. |
| National Institute of Korean Language (NIKL) 2025 corpora | Application / agreement required; do not ingest yet | Official FAQ limits use to approved research and technology-development purposes. The exact collection agreement must be checked for model pretraining, retaining trained weights, service deployment, and any commercial use. |
| FineWeb-2 Korean | Blocked pending rights review | ODC-By 1.0 covers database rights; individual web-page rights and Common Crawl terms remain separate. |
| FineWeb English | Blocked pending rights review | Same database/content-rights distinction as FineWeb-2 Korean. |
| Cosmopedia v0.1 | Candidate only | Dataset card labels Apache-2.0, but the corpus is synthetic and seed-source provenance, quality, and factual reliability need review. |
| Sage Pretrain Corpus | Candidate only | Mixed per-record licenses require row-level provenance and allowlist filtering. |
| WanJuan-Korean | Candidate only | Dataset-level CC BY label is not enough; schema, source provenance, spam, and privacy checks are still required. |
| Wikimedia Wikipedia XML dumps (Korean + English) | Explicit candidate; staging only; CC BY-SA excluded from trainer | Official dump processor exists and now records revision/dump provenance. Per-page imported-content review and a documented decision on ShareAlike obligations for trained weights are still required. |\n| Stack Exchange public data | Excluded by current trainer allowlist | Version-dependent CC BY-SA terms require post/revision-level attribution and ShareAlike review. |

## NIKL application questions to resolve

Before requesting or staging any NIKL collection, obtain written clarification from NIKL or the collection administrator on:

1. Whether the approved purpose explicitly includes training a Transformer language model using next-token prediction.
2. Whether the resulting model weights may be retained after the corpus-use agreement expires.
3. Whether the model may be operated as an online AI service, and whether any commercial use is allowed.
4. Whether the model, checkpoints, tokenizer, or training artifacts may be distributed.
5. Which attribution, deletion, audit, and public-output approval requirements apply.
6. Whether each selected collection has additional source-specific terms.

The public FAQ describes approved research / technology-development use and prior approval for public outputs containing corpus text. It does not, by itself, answer all model-training and deployment questions.

Official pages:
- FAQ: https://kli.korean.go.kr/boards/faqList.do?lang=ko
- Application: https://kli.korean.go.kr/main/requestMain.do?lang=ko
- 2025 corpus catalogue: https://kli.korean.go.kr/main/requestMain.do?lang=ko

## Current Common Voice intake

- Archive: `cv-corpus-27.0-2026-09-11`
- Bundled README: Creative Commons Zero (CC0)
- Parsed records: 623 train, 514 dev, 579 test (test excluded)
- Local staging: 12 train blocks, 9 validation blocks
- Candidate BPE tokens: 10,721 train; 9,048 validation
- Original retrieval date: unknown; do not substitute upload date for source retrieval date.
- `license_reviewed` and `approved_for_training` remain false until final human review.
- This corpus is supplemental only and is not sufficient for a general 10,000-step pretraining run.

## Training gate

Do not start the official 10,000-step run until:
- enough general-domain Korean and English data is available;
- each source has a pinned version, retrieval date, license evidence, attribution, and source-level provenance;
- data filtering, privacy checks, deduplication, and train/validation/test separation pass;
- the final manifest has explicit human review and approval;
- validation data remains excluded from BPE merge fitting and gradient updates.

The existing production Quantum model and checkpoints must remain untouched. Candidate training stays isolated.


## Wikimedia decision and required gate

Decision: **Use official Korean and English Wikimedia dumps as a candidate corpus source, but do not train on them yet.** The offline importer may stage pages for review. It must not auto-approve pages or change the trainer's license allowlist.

Before any Wikimedia page enters the training manifest:

1. Pin the exact dump filename, language, dump SHA-256, and actual download date.
2. Preserve page ID, revision ID, article URL, revision-history URL, title, attribution, license URL, and the text-normalization/modification note.
3. Review the exact revision and page footer for imported content, third-party notices, exclusions, and differing license terms.
4. Remove unsuitable, low-quality, duplicate, privacy-sensitive, or non-article content.
5. Keep train and validation splits separate and exclude the fixed evaluation set.
6. Obtain a documented project-level legal/policy decision on whether training on CC BY-SA text and retaining/distributing/serving resulting model weights triggers ShareAlike or other obligations. CC BY-SA source licensing does not by itself answer that model-weight question.
7. Only after approval, implement the required attribution and license compliance for the corpus, checkpoints, and any distributed/served model artifacts. Then update the trainer allowlist through a separately reviewed change.

Official references:
- Wikimedia Terms of Use: https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use
- Wikimedia reuse guidance: https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_Trademark_Policy (check current policy links before relying on them)
- CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/

The current importer records the original dump retrieval date as unknown unless supplied by a later reviewed manifest update; it records the processing date separately. The training engine still rejects `CC-BY-SA-4.0`.


## Current Wikimedia dump format

Use Wikimedia's newer monthly `mediawiki_content_current` export for current page revisions. The legacy XML dump path is formally being replaced. Download all XML parts listed by the official `SHA256SUMS` file, verify each part, and pass all parts to the importer in one run. Avoid the full-history export for the initial corpus. The export uses a consistent first-of-month UTC cutoff and is published after a delay for revision suppression/deletion processing.

Official technical documentation: https://wikitech.wikimedia.org/wiki/MediaWiki_Content_File_Exports


## Formal CC BY-SA 4.0 assessment for Quantum (2026-11-22)

**Review scope:** copyright-license analysis for text-only use of Wikipedia article revisions in JUCKEX Quantum next-token pretraining. This is an engineering/policy assessment, not a legal opinion from qualified counsel. It does not decide copyright exceptions under Korean, US, or other national law.

### Executive decision

**Decision: CONDITIONAL CANDIDATE — NOT APPROVED FOR THE TRAINING MANIFEST.**

CC BY-SA 4.0 does not prohibit commercial reuse. It grants broad rights to reproduce, share, and adapt licensed material, subject to attribution and ShareAlike conditions when the relevant material or an adaptation is publicly shared. Creative Commons also states that CC-licensed material can be used in AI development where copyright permission is required, provided the applicable license conditions are respected.

However, the license does not clearly answer whether trained neural-network weights are themselves an adaptation of the training texts. That question depends on the model, memorization/retention, output behavior, applicable law, and the facts of distribution. The project must not assume either that all model weights automatically inherit CC BY-SA or that weights are categorically outside it.

For this reason, Wikipedia may be staged for audit, but the existing trainer allowlist remains unchanged and rejects `CC-BY-SA-4.0`. Do not start a run containing Wikipedia text until a qualified copyright review or a documented project policy decision resolves the model-weight and deployment questions.

### Findings by use

| Activity | Assessment | Project rule |
|---|---|---|
| Download and privately store official dump parts | Generally permitted under the license where permission is required, subject to applicable terms and source integrity | Keep exact dump, checksum, retrieval date, and access logs |
| Extract text and privately preprocess for training | CC 4.0 expressly permits private reproduction, extraction, reuse, and creation of adapted databases for text/data mining where permission is required | May stage only; do not auto-approve |
| Train a model privately | CC permits AI development uses where copyright permission is required, subject to conditions that become relevant to the use and sharing | Conditional; do not train until the project resolves model-weight treatment |
| Keep model weights private and operate an inference service | No automatic conclusion that weights are adapted material; public outputs may raise separate questions if they reproduce protected expression | Require legal review of model design, memorization controls, outputs, and service model |
| Publicly distribute raw Wikipedia text or a substantially copied/modified corpus | Attribution required; modified/adapted material and qualifying adapted databases can trigger ShareAlike | Do not distribute without compliant CC BY-SA notices and share-alike handling |
| Publicly distribute trained weights/checkpoints | Unsettled whether weights are adapted material; cannot infer the answer from the dataset license alone | Block until written legal/policy decision |
| Publish generated answers | Ordinary factual summaries are not automatically an adaptation; substantial or near-verbatim reproduction of a protected passage may be | Add output similarity/memorization tests and source attribution where source text is used |

### License obligations that are clear

1. **Attribution:** where the licensed text or an adaptation is shared, retain creator/attribution information supplied with it, copyright/license notices and a URI to the source where reasonably practicable. Wikimedia accepts attribution through a link/URL to the article, whose history lists contributors, or an equivalent stable copy / author list.
2. **Modification notice:** state that text was modified and describe the transformations. Quantum's importer strips/normalizes wikitext; this must be recorded as a modification.
3. **ShareAlike:** if an adaptation of CC BY-SA material is publicly shared, license that adaptation under CC BY-SA 4.0 or a compatible license and do not impose incompatible downstream restrictions.
4. **Database rights:** CC BY-SA 4.0 also covers applicable sui generis database rights. If a substantial portion of a protected database is included in another publicly shared database, the adapted database may itself be subject to ShareAlike.
5. **Third-party material:** Wikimedia pages can include imported text with additional attribution requirements or different license terms. Page footer, history, and discussion/source notices must be reviewed; do not assume every byte in a dump is uniformly CC BY-SA.
6. **Non-text media:** this pipeline is text-only. Images, audio, and other media are excluded and must not be imported under the article text's license assumption.

### AI-training and model-weight uncertainty

Creative Commons' FAQ says CC-licensed works may be used in development of new technologies, including AI, where copyright permission is required, provided license conditions are respected. Its text/data-mining guidance distinguishes private mining from publicly sharing mining results or mined data; if a publicly shared result is an adaptation of SA material, the same/compatible SA license applies.

That guidance does **not** expressly classify neural-network parameters, checkpoints, tokenizers, or model outputs as adaptations. Therefore:
- Do not assert that CC BY-SA necessarily forces all Quantum weights to be open-licensed.
- Do not assert that model weights are automatically exempt.
- Treat weight distribution, hosted inference, and generated output as separate questions for counsel.
- A license label on the model card cannot cure a rights issue if the artifact is in fact an adaptation; likewise, do not apply CC BY-SA to unrelated project code or independently created model components without a basis.

### Required compliance controls before any training approval

- Pin exact monthly Wikimedia export date, all filenames, per-file SHA-256 values, combined digest, and actual download timestamp.
- Keep page ID, revision ID, canonical article URL, revision-history URL, title, language, and contributor attribution route.
- Inspect article footer/history/discussion and imported-content notices; exclude pages with incompatible or unclear rights.
- Exclude images, audio, files, and non-text media from this text-only pipeline.
- Filter personal information, sensitive personal data, biographies of living persons where risk is elevated, spam, vandalism, redirects, and low-quality/short pages. Human review remains necessary.
- Preserve a machine-readable per-document license and attribution manifest.
- Keep source corpus, validation corpus, and fixed evaluation data separate.
- Provide a public attribution page if source text or an adaptation is shared; link each article/revision and the CC BY-SA 4.0 license, describe text normalization, and provide a compliant copy/license notice for distributed adapted text.
- Add memorization and near-verbatim output tests before exposing the candidate model to users.
- Do not distribute the corpus or checkpoints until their applicable license terms are resolved.

### Final engineering decision

The current implementation is correct to keep `CC-BY-SA-4.0` outside `ALLOWED_LICENSES`. The Wikimedia importer is an intake/audit tool only. No change to the trainer allowlist is authorized by this review. A future enablement change must be a separate reviewed commit after qualified legal advice or an explicit documented risk decision by the project owner.

### Primary sources

- CC BY-SA 4.0 legal code: https://creativecommons.org/licenses/by-sa/4.0/legalcode
- Creative Commons FAQ — text/data mining and AI: https://creativecommons.org/faq/#can-i-conduct-textdata-mining-on-a-cc-licensed-database
- Wikimedia Foundation Terms of Use, section 7 (licensing, attribution, imported text, reuse and modifications): https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use
