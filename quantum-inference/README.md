# JUCKEX Quantum Candidate — isolated inference service

이 서비스는 JUCKEX Production과 분리된 실험용 Candidate입니다. 이 브랜치의 코드 변경은 Production Worker·D1·KV·Render 서비스를 변경하지 않습니다.

## 런타임

- Python 3.12 / FastAPI / PyTorch CPU
- Render 서비스: `juckex-quantum-candidate`
- 실행 명령: `cd quantum-inference && QUANTUM_CHECKPOINT_PATH=../bpe_candidate_checkpoint.pt uvicorn app:app --host 0.0.0.0 --port $PORT`
- 빌드 명령: `pip install -r quantum-inference/requirements.txt`
- 상태: `GET /` 또는 `GET /health`
- 추론: `POST /v1/generate` (JSON: `{"prompt":"질문" }`)

## 현재 체크포인트 상태

체크포인트 파일은 저장소에 평문으로 커밋하지 않습니다. GitHub Releases에는 `quantum-candidate-45010825022` 태그의 암호화된 `quantum-candidate.pt.enc` 자산이 존재합니다. 다만 릴리스 자산이 존재한다는 사실만으로 Render 인스턴스에서 복원에 성공했다고 볼 수는 없습니다.

- `/health`의 `model_loaded`와 `generation_ready`가 모두 `true`여야 추론 준비 상태입니다.
- `model_parameters`, `model_architecture`, `checkpoint_source`, `checkpoint_load_error`를 실제 런타임에서 확인합니다.
- 체크포인트 복원 실패 또는 누락 시 `/v1/generate`는 정상 생성 대신 HTTP 503으로 거부합니다.
- 비밀키나 복호화된 체크포인트를 공개 저장소에 커밋하지 않습니다.

Render에서는 `QUANTUM_CHECKPOINT_RELEASE_TAG`에 고정된 릴리스 태그와 Secret `QUANTUM_CHECKPOINT_ENCRYPTION_KEY`가 필요합니다. 로컬 파일 방식이면 `QUANTUM_CHECKPOINT_PATH`가 Candidate 체크포인트 파일을 정확히 가리켜야 합니다. 이 값은 채팅이나 저장소에 공개하지 않습니다.

## 모델 규모 및 품질

저장소에 있는 2026-11-22 평가 보고서는 당시 Candidate 학습 산출물을 약 **1,089,280개 파라미터**로 기록합니다. 이 보고서의 독립 평가 생성 예시에는 질문과 무관하거나 문장이 불완전한 출력이 다수 포함되어 있습니다. 보고서의 문자당 비트 수(BPC)는 다음 토큰 예측 지표이지 사실 정확도나 대화 품질 점수가 아닙니다.

별도 Candidate Render 실행(2026-11-23)의 held-out 평가에서는 31개 사례 중 30개가 실제 BPE Candidate 모델 생성 호출이었고, 1개는 승인 Q&A 검색 응답이었습니다. 생성 30건 모두에서 공백 단위 한 글자 토큰 분절 패턴이 감지되었습니다. 따라서 해당 체크포인트는 한국어 사용자 대상 생성 품질 기준을 통과하지 못했습니다. 실제 기록은 [evaluation/BASELINE_2026-11-23.md](evaluation/BASELINE_2026-11-23.md)에 보관합니다.

이 수치는 현재 Render 인스턴스에서 로드된 모델의 실측치가 아닙니다. 실제 모델 규모는 체크포인트가 연결된 뒤 `/health`가 반환하는 `model_parameters`로 확인해야 합니다.

## Candidate BPE 공백 경계 정책

Candidate의 `bpe_tokenizer.py` 학습기는 병합 쌍을 공백·탭·줄바꿈을 가로질러 학습하지 않습니다. 비공백 문자열 구간 안에서만 반복 쌍을 학습하고, 각 공백 문자는 별도 구간으로 유지합니다. 따라서 단어 사이 공백이나 줄바꿈을 우연히 하나의 서브워드 토큰으로 합치는 학습을 방지합니다. 이는 토크나이저의 분절 정책이며 한국어 형태소 분석이나 의미 단위 분절을 제공하지 않습니다. 실제 모델 입력에서의 이점은 독립 평가 코퍼스의 토큰 수, round-trip, 생성 결과를 통해 별도로 검증해야 합니다.

## 토크나이저 진단

`tokenizer_diagnostics.py`는 학습/검증 말뭉치에 대해 다음을 별도로 기록합니다.

- Unicode 코드포인트 수와 실제 인코딩 토큰 수
- 미등록(UNK) 토큰 수와 비율
- 여러 코드포인트를 합친 토큰 사용 비율
- 코드포인트 기준 토큰 감소율과 평균 토큰 길이
- 어휘 내 다중 코드포인트 병합 항목 수
- 완성형 한글 음절과 분해 자모 코드포인트 수
- 한글 음절을 포함하는 토큰 중 한 음절만 포함한 토큰 비율
- 한글 음절 포함 토큰당 평균 음절 수
- 전체 원문 왕복 변환 성공 여부

이 수치는 토크나이저의 범위·압축 동작과 한글 음절 분절 양상을 진단하기 위한 것입니다. 한글 음절 지표는 형태소나 단어 경계를 판별하지 않으며, 자연스러운 문장 생성, 사실 정확도 또는 추론 능력을 의미하지 않습니다. 특히 낮은 토큰 수나 높은 압축률만으로 모델 품질이 개선되었다고 판정하지 않습니다.

새 학습 보고서에는 학습 세트와 검증 세트의 진단 결과가 각각 포함됩니다. 두 세트의 통계를 섞지 않아 검증 자료의 토큰 분포를 별도로 살펴볼 수 있습니다.

## 생성 출력 토큰 진단

Candidate의 `POST /v1/generate` 모델 생성 응답에는 `generation_diagnostics`가 포함됩니다.

- 생성 토큰 수와 UNK 토큰 수/비율
- 한 코드포인트 토큰 사용 비율
- 한글 음절을 포함하는 생성 토큰 수와 한 음절 토큰 비율
- 한글 음절 포함 토큰당 평균 음절 수
- 한 글자 토큰 분절 의심 여부

한 글자 토큰 비율이 높고 출력이 충분히 길면 분절 의심 플래그를 표시하고, 응답 경고에도 이를 알립니다. 이는 기존 평가에서 관찰된 분절 문제를 런타임에서 추적하기 위한 진단입니다. 한글 전용 지표는 완성형 음절을 기준으로 하며, 분해 자모를 형태소나 음절 단위로 자동 결합하지 않습니다. 플래그는 문장 자연스러움이나 정답 여부를 자동 판정하지 않으며, 모델 품질 점수로 사용하지 않습니다. 승인 Q&A 검색 응답과 모델 생성 응답의 모드 구분은 그대로 유지됩니다.

## 평가 데이터

`evaluation/benchmark_v1.json`은 일상 대화, 역사, 과학, 수학, 경제, 사회, 지리, 기술, 프로그래밍, 언어, 논리, 복합 질문, 긴 문맥, 사실 확인, 불확실성 등 15개 영역의 held-out 평가 사례를 포함합니다. `training_eligible: false`로 표시되며 학습 데이터로 사용하면 안 됩니다.

평가 결과는 정확성·근거성·문맥 일관성·불확실성 표현을 사례별로 기록해야 합니다. 기준 답안과 일치하는 단어가 있다는 이유만으로 정답 처리하지 않습니다. 최신 통계와 시점 민감 정보는 신뢰할 수 있는 최신 출처를 확인해야 합니다.

로컬 검증:
```bash
cd quantum-inference
python -m unittest -v test_tokenizer_diagnostics.py test_bpe_tokenizer_integrity.py
python -m unittest discover -s . -p 'test_*.py' -v
```

## 안전 경계

- Candidate 전용이며 Production 승격은 자동으로 하지 않습니다.
- Candidate 학습과 추론은 Production 모델 가중치를 수정하지 않습니다.
- 미검증 위키백과 문서는 사실 지식으로 취급하지 않습니다.
- 위키백과 자료를 사용할 때 CC BY-SA 4.0의 저작자 표시 및 적용되는 동일조건변경허락 조건을 보존해야 합니다.
- 체크포인트 연결, 추론 테스트, 독립 평가가 모두 확인되기 전에는 배포 성공이나 모델 준비 완료로 보고하지 않습니다.
- 평가 사례와 모델 답변은 명시적 승인과 별도 검토 없이 학습 코퍼스에 편입하지 않습니다.


## Wikipedia Candidate export contract

`export_wikipedia_corpus.py` downloads the private Candidate corpus from the isolated Worker API. It accepts only the `manual` and `unverified` trust classes and validates every document before atomically replacing the destination file.

- Require string page identity, title, body, revision ID, hash, status, and attribution metadata.
- Accept only canonical HTTPS URLs on `ko.wikipedia.org`.
- Verify the CC BY-SA 4.0 attribution contains the page title, exact page URL, license name, and license URL.
- Recompute SHA-256 from the normalized document body and reject mismatches.
- Require a timezone-aware ISO-8601 `reviewed_at` timestamp for the manual corpus; unverified records must not carry a review timestamp.
- Keep manual approvals separate from pending-review records. An approval marker and timestamp do **not** establish a named reviewer's identity or independently prove factual correctness.
- Create owner-only output files and preserve an existing destination if any page or pagination response fails validation.

The crawler generates page-specific attribution when collecting or refreshing a Wikipedia page. Export validation is not a substitute for legal review, factual verification, or the separate training-data provenance gate. Wikipedia retrieval updates do not change Quantum model weights.


## Candidate generation report integrity for human review

Human-review templates exported by `export_generation_review_template.py` include `report_sha256`, a SHA-256 fingerprint of the complete generation report serialized as deterministic UTF-8 JSON (sorted keys, compact separators, no non-finite numbers). The review summarizer recomputes the fingerprint and rejects a review document if it is missing or references a different report, even when the benchmark ID is unchanged.

This binds the ratings to the exact report content reviewed and helps prevent accidental mixing of separate Candidate runs. It does **not** authenticate the reviewer, prove the truth of ratings, or authorize training or Production promotion. The summary continues to set `reviewer_identity_verified=false`, `training_eligible=false`, and `promotion_approved=false`.


## Provenance review and reviewer identity boundary

The provenance audit distinguishes metadata completeness from proof of reviewer identity. A non-empty reviewer identifier and timezone-aware review timestamp are required metadata, but they are not cryptographic authentication. Audit output therefore reports `reviewer_identity_verified=false`, `reviewer_identity_verification_required=true`, and explains that `training_release_eligible` is a metadata-completeness gate only. The review-application tool also reports this limitation explicitly. Do not interpret a completed metadata audit as proof that a named person authenticated or that source claims are true.


## Candidate decoding and seed-result input validation

The Candidate n-gram repetition guard accepts only integer token IDs within the current logits vocabulary before building repetition bans. Invalid, boolean, negative, out-of-range, or non-integer history values are rejected rather than causing an unhandled set-membership failure or masking an unintended token.

Wikipedia seed completion summaries also require non-negative integer counters (booleans, floats, numeric strings, and negative values are rejected). Retrieval pass counts cannot exceed the total. This prevents malformed workflow/API values from being coerced into a misleading `completed` status. These checks validate report integrity; they do not establish retrieval relevance or factual correctness.


The Wikipedia seed reuse shortcut applies the same strictness to cached index metadata: document counts and required thresholds must be actual integers (not booleans or floats), and requested/available title collections must be lists or tuples of strings. Malformed metadata causes a fresh seed path instead of incorrectly treating an index as reusable.


## Retrieval content-integrity enforcement

At read time, knowledge retrieval now requires a numeric Wikipedia revision ID and a lowercase SHA-256 digest that exactly matches the stored UTF-8 text. Missing revision metadata, malformed digests, and text/hash mismatches are excluded from model context and citations. The generation-source serializer repeats this integrity check as a second boundary, so callers cannot bypass retrieval filtering by passing a malformed row directly.

The API reports `rejected_integrity_rows` separately from `rejected_source_rows`. This detects accidental or unauthorized local database edits; it does not prove the upstream article is factually correct or authenticate who reviewed it.

## Retrieval source-attribution enforcement

The SQLite knowledge retrieval path validates the source metadata again at read time before returning any document. A row is eligible only when its URL is a canonical HTTPS Korean Wikipedia page and its attribution includes the exact title, exact URL, CC BY-SA 4.0 name, and canonical license URL. Rows that fail this check are excluded even if their lexical relevance score is high.

The response reports `rejected_source_rows`. If every otherwise relevant row fails source validation, the status is `source_metadata_invalid`; a valid subset can still be returned with the normal FTS/fallback status. This read-time check is defense in depth and does not replace import-time validation, license review, or factual review.

## Retrieval recall at K

Benchmark case validation also applies input-size limits: IDs up to 128 characters, domains up to 80, questions up to 2,000, at most 20 expected titles per case, and titles up to 256 characters. This bounds malformed or accidental oversized evaluation inputs before retrieval is invoked. The retrieval-only benchmark also measures coverage when a case expects multiple relevant documents. Each answerable case reports `expected_title_count`, `relevant_retrieved`, and `recall_at_k`; the overall and per-domain reports expose `mean_recall_at_k` averaged over answerable cases only. They also expose `micro_recall_at_k`, which divides all relevant retrieved titles by all expected titles across answerable cases; unlike the macro mean, this weights cases with more expected documents more heavily. Unanswerable cases use `null` for recall because recall is not applicable when no expected documents exist. A mixed report excludes those null cases from the recall mean while still counting them in safe-abstention metrics.

Recall is based on the benchmark's explicitly labeled expected titles, not an automatic semantic relevance judgment. It is a retrieval coverage measure, not a model answer quality score.

## Retrieval noise metrics

The retrieval-only benchmark now reports relevant and irrelevant returned-document counts per case, overall retrieval noise rate, and per-domain retrieval noise rate. For answerable cases, an expected-title hit does not hide unrelated extra results: every returned title outside the case's expected-title set contributes to retrieval noise. For unanswerable cases, any returned document remains a false-positive retrieval and also contributes to noise. The rate is `null` when no documents were returned, rather than being reported as zero noise without evidence.

These metrics evaluate retrieval behavior only. They do not establish factual correctness, semantic relevance beyond the benchmark's expected-title labels, or model answer quality.

## Retrieval benchmark: safe abstention and anti-inflation checks

The retrieval-only benchmark now reports answerability separately from ordinary title hits. Cases with an empty `expected_titles` list are explicitly unanswerable cases: returning any document is counted as a false-positive retrieval, while returning no documents is a correct abstention. Reports include `unanswerable_cases`, `false_positive_retrievals`, and `safe_abstention_rate`, with per-domain equivalents.

The benchmark rejects non-string case identity/question/domain fields, invalid integer limits, malformed retrieved titles, and duplicate normalized retrieved titles. Duplicate results are rejected rather than allowed to inflate precision. These metrics evaluate retrieval behavior only; they do not establish factual correctness, model generation quality, or training eligibility. A dedicated Candidate-only GitHub Actions workflow runs this regression suite without contacting the model or Production services.


## Candidate source traceability

The authenticated `POST /admin/knowledge/search` diagnostic now returns a source trace for every eligible retrieval result: the Wikipedia page ID, revision ID, stored UTF-8 content SHA-256, canonical page URL, license and attribution, lexical relevance score, and a short excerpt. This makes it possible to identify the exact imported page revision and verify whether its stored text still matches the recorded digest.

The trace endpoint is retrieval-only: it does not invoke Quantum, change model weights, or prove that a retrieved passage supports a generated claim. The relevance score is a lexical ranking signal, not a probability or factual-confidence score. The SHA-256 digest detects content changes relative to the stored revision record; it does not establish upstream truth or reviewer identity. The endpoint remains protected by Candidate admin authentication.


### Citation suppression when the model context is truncated

Candidate generation suppresses Wikipedia citations when the retrieved knowledge context is truncated to fit the model's token window. The current token-budget helper keeps the tail of the serialized context; without per-passage token-span accounting, the API cannot prove which complete source passages remain. In that case the response returns an empty `knowledge_sources` list, preserves the original `retrieved_source_count`, sets the effective `source_count` to zero, and reports `citations_suppressed_reason: "context_truncated"`. This is intentionally conservative: it avoids presenting retrieved-but-not-provably-included documents as citations. It does not establish that any untruncated source semantically supports the generated answer.


### Candidate generation context-window diagnostics

The model-generation response includes a `context_diagnostics` object calculated from the active tokenizer output and the configured model block size. It reports the block size, raw question-text tokens, formatted question-prompt tokens (including the prompt markers), retrieved context tokens before truncation, context tokens actually included, dropped context tokens, total prompt tokens, remaining generation capacity, truncation state, and the current `keep_tail` context policy. These values are token-budget accounting, not a measure of semantic coverage, factual support, or answer quality. The remaining generation capacity describes unused input-window positions at prompt construction time; the request's separate `max_new_tokens` limit and generation loop determine actual output length.


### Candidate runtime output-shape diagnostics

The generation response also includes `output_shape_diagnostics` from `generation_output_diagnostics.py`. It reports visible whitespace-token fragmentation and repeated three-token sequence patterns. A repetition warning requires at least 12 lexical tokens, at least two repeated trigram occurrences, and a repeated-occurrence ratio of at least 0.20. These conservative thresholds are anomaly flags only; they do not automatically reject an answer or measure fluency, factuality, correctness, or overall quality. The same helper is used by the held-out benchmark and runtime API so both paths report the same diagnostic definitions.

## Provenance filtering before the final retrieval limit

The SQLite retrieval path ranks its bounded candidate batch (up to 100 FTS rows or 100 fallback rows), validates source attribution and content integrity for those ranked candidates, and only then applies the caller's requested result limit (1–10). This prevents a small number of high-scoring but invalid rows from occupying the entire result window and hiding valid documents immediately below them.

The standalone lexical ranking helper accepts a bounded limit up to 100 for this internal candidate-batch use. The public index-search result limit remains 1–10. If every candidate is rejected, the response reports the relevant source-metadata or content-integrity failure state; mixed source and integrity failures use `document_metadata_invalid`. Rejection counters remain separate.

This improves candidate selection under malformed-index conditions. It does not improve semantic relevance, establish factual correctness, or prove that a returned passage supports a generated claim.



## Korean retrieval query normalization

The lexical query normalizer preserves a conservative set of common Korean nouns whose final syllables overlap with particle strings (for example, 국가, 사과, 회의, 전문가, 종이, 고양이, 정의, 성과, and 도로). It removes at most two recognized outer endings, including common compound postpositions such as 로부터, 에서부터, and 에게서. Examples include 국가의, 국가에서는, 국가로부터, 회의에서부터, 전문가에게서, 종이는, 정의의, and 성과가. This bounded rule also handles forms such as 회의에서의 without becoming an unbounded stemmer. It is a targeted ambiguity guard, not a full Korean morphological analyzer; uncommon words and other grammatical endings still require corpus-driven evaluation.

## BPE serialized merge boundary validation

The isolated inference-side BPE tokenizer rejects serialized merge rules when either input token contains whitespace. Training already keeps spaces, tabs, and line breaks as independent segments; constructor validation now enforces the same invariant when loading a tokenizer artifact, so a manually edited or malformed payload cannot reintroduce merges across word or line boundaries. This is a tokenizer-format integrity rule, not a Korean morphology feature or a language-quality improvement.

## Retrieval benchmark iterable bounds

The retrieval benchmark validates iterable inputs without first materializing the entire source. It consumes at most `max_cases + 1` items (default maximum: 500); the extra item is used only to detect an over-limit dataset. This also bounds generator inputs and prevents an unbounded iterator from being exhausted before the case-count check. Non-iterable inputs fail with a clear validation error. This is an input-resource safeguard and does not change retrieval metrics or model behavior.

## Retrieval query resource bounds

The Korean retrieval query normalizer rejects questions longer than 2,000 Unicode characters before tokenization and retains at most 64 unique lexical terms in first-occurrence order. The FTS query builder applies its stricter caller-supplied term limit (10 by default). This bounds regex tokenization, ranking work, and fallback SQL parameter expansion for direct library callers as well as API callers. Oversized queries fail explicitly instead of being silently accepted; term-heavy queries retain their earliest distinct terms. This is an input/resource bound, not a relevance or answer-quality improvement.


Domain-alias expansion is subject to the same 64-unique-term ceiling as the original query. Alias terms are appended only when budget remains; a query already using the full lexical-term budget does not gain extra alias terms. This closes a secondary expansion path that could otherwise exceed the documented retrieval work bound. It is a resource-bound guarantee, not a search-relevance improvement.
