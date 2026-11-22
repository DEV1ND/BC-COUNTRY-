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

체크포인트 파일은 저장소에 커밋하지 않습니다. 현재 Candidate 브랜치의 파일 목록에는 `bpe_candidate_checkpoint.pt`가 없고 GitHub Releases에도 암호화된 Candidate 체크포인트 릴리스가 확인되지 않았습니다. 따라서 Render의 실행 파일 경로만으로 모델이 로드된다고 볼 수 없습니다.

- 체크포인트가 없으면 `/health`는 `model_loaded: false`와 파라미터 수 0을 반환합니다.
- 이 상태에서 `/v1/generate`는 정상 답변을 생성하지 않고 HTTP 503으로 거부합니다.
- 실제 추론 검증은 비공개 Candidate 체크포인트를 안전하게 연결한 뒤 수행해야 합니다.
- 비밀키나 체크포인트를 공개 저장소에 커밋하지 않습니다.

`QUANTUM_CHECKPOINT_PATH`는 비공개로 마운트된 Candidate 체크포인트 경로를 가리켜야 합니다. 암호화된 GitHub Release를 사용할 경우 `QUANTUM_CHECKPOINT_RELEASE_TAG`와 Render Secret의 `QUANTUM_CHECKPOINT_ENCRYPTION_KEY`가 필요합니다. 이 값은 채팅이나 저장소에 공개하지 않습니다.

## 모델 규모 및 품질

저장소에 있는 2026-11-22 평가 보고서는 당시 Candidate 학습 산출물을 약 **1,089,280개 파라미터**로 기록합니다. 이 보고서의 독립 평가 생성 예시에는 질문과 무관하거나 문장이 불완전한 출력이 다수 포함되어 있습니다. 보고서의 문자당 비트 수(BPC)는 다음 토큰 예측 지표이지 사실 정확도나 대화 품질 점수가 아닙니다.

이 수치는 현재 Render 인스턴스에서 로드된 모델의 실측치가 아닙니다. 실제 모델 규모는 체크포인트가 연결된 뒤 `/health`가 반환하는 `model_parameters`로 확인해야 합니다.

## 평가 데이터

`evaluation/benchmark_v1.json`은 15개 영역, 30개 사례로 구성된 별도 한국어 평가 벤치마크입니다. `training_eligible: false`로 표시되며 학습 데이터로 사용하면 안 됩니다.

로컬 검증:
```bash
cd quantum-inference
python -m unittest -v tests/test_evaluation_benchmark.py
python -m unittest discover -s tests -v
```

평가 결과는 정확성·근거성·문맥 일관성·불확실성 표현을 사례별로 기록해야 합니다. 기준 답안과 일치하는 단어가 있다는 이유만으로 정답 처리하지 않습니다. 최신 통계와 시점 민감 정보는 신뢰할 수 있는 최신 출처를 확인해야 합니다.

## 안전 경계

- Candidate 전용이며 Production 승격은 자동으로 하지 않습니다.
- Candidate 학습과 추론은 Production 모델 가중치를 수정하지 않습니다.
- 미검증 위키백과 문서는 사실 지식으로 취급하지 않습니다.
- 위키백과 자료를 사용할 때 CC BY-SA 4.0의 저작자 표시 및 적용되는 동일조건변경허락 조건을 보존해야 합니다.
- 체크포인트 연결, 추론 테스트, 독립 평가가 모두 확인되기 전에는 배포 성공이나 모델 준비 완료로 보고하지 않습니다.
