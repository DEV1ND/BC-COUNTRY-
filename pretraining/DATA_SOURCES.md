# Quantum pretraining corpus source register

This register is a cautious intake plan, not evidence that any dataset has been downloaded, approved, or trained on. Each exact dataset/version still needs its own review.

## Admission rule

The training script currently accepts only `CC0-1.0` and `CC-BY-4.0` in the manifest. The source's actual license must be verified at dataset/file level. Record the source URL, retrieval date, license URL/version, attribution, data version, and reviewer decision. A license label copied from a third-party catalog is not sufficient by itself.

Do not commit raw corpus files, account credentials, private data, or datasets whose terms prohibit this use or redistribution. Keep source data in a controlled local/R2 training workspace with access limited to authorized project maintainers.

## Candidate sources and current disposition

| Source | Intended contribution | Disposition | Reason / required action |
|---|---|---|---|
| Wikimedia / Korean Wikipedia dump | Korean encyclopedic prose and broad topic coverage | HOLD — license review | Most Wikipedia text is CC BY-SA 4.0; verify the exact dump and attribution/share-alike obligations. Current trainer deliberately rejects CC BY-SA pending a separate review. |
| AI Hub open datasets | Korean language coverage and domain-specific text | REQUEST / REVIEW PER DATASET | AI Hub requires application/identity and purpose disclosure for downloads; NIA attribution is required, some datasets have separate provider rules, and data must not be passed to unauthorized third parties. Confirm the individual dataset's terms before use. |
| National Institute of Korean Language, 모두의 말뭉치 | High-quality Korean written/spoken language | REQUEST / REVIEW PER CORPUS | Access is by application and approved purpose. Its FAQ says output publication may require prior approval and must not contain source corpus text. Confirm exact corpus-specific agreement and whether model training/weights and JUCKEX's intended use are allowed. |
| User-owned or commissioned original text | Domain-balanced original material | ELIGIBLE AFTER RIGHTS CHECK | Retain author/rights-holder permission and explicit model-training scope; remove personal and confidential content. |
| Public-domain works | Historical and literary language | REVIEW PER WORK/JURISDICTION | Verify public-domain status in relevant jurisdictions, edition/editorial rights, and source quality. |

## Official policy references

- AI Hub data-use policy: https://aihub.or.kr/intrcn/guid/usagepolicy.do
- National Institute of Korean Language corpus portal: https://kli.korean.go.kr/
- National Institute of Korean Language FAQ: https://kli.korean.go.kr/boards/faqList.do
- Wikimedia content reuse and attribution: https://www.mediawiki.org/wiki/Wikimedia_APIs/Content_reuse

## Intake workflow

1. Select a specific dataset and version; do not approve a provider wholesale.
2. Save its official terms, license text, attribution requirements, and retrieval date.
3. Confirm model-training use, output/weight distribution, commercial use (if relevant), and geographic restrictions.
4. Inspect privacy, consent, personal information, and third-party content.
5. Deduplicate documents and assign train/validation splits before tokenization.
6. Keep the final evaluation set isolated and never add benchmark questions to training.
7. Have a human reviewer record approval before adding any file path to `manifest.jsonl`.

## Current state

- No raw dataset is staged by this PR.
- No dataset is approved by this register alone.
- No pretraining has been run.
- The 10,000-step run remains blocked until enough reviewed corpus text is staged and the manifest is complete.
