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

## 토크나이저 진단

`tokenizer_diagnostics.py`는 학습/검증 말뭉치에 대해 다음을 별도로 기록합니다.

- Unicode 코드포인트 수와 실제 인코딩 토큰 수
- 미등록(UNK) 토큰 수와 비율
- 여러 코드포인트를 합친 토큰 사용 비율
- 코드포인트 기준 토큰 감소율과 평균 토큰 길이
- 어휘 내 다중 코드포인트 병합 항목 수
- 전체 원문 왕복 변환 성공 여부

이 수치는 토크나이저의 범위·압축 동작을 진단하기 위한 것입니다. 자연스러운 문장 생성, 사실 정확도 또는 추론 능력을 의미하지 않습니다. 특히 낮은 토큰 수나 높은 압축률만으로 모델 품질이 개선되었다고 판정하지 않습니다.

새 학습 보고서에는 학습 세트와 검증 세트의 진단 결과가 각각 포함됩니다. 두 세트의 통계를 섞지 않아 검증 자료의 토큰 분포를 별도로 살펴볼 수 있습니다.

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
