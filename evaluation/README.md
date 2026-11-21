# QUANTUM EVALUATION v1

운영 Quantum과 분리된 한국어 평가 전용 세트 및 실행기입니다.

## 정식 평가 데이터
- `quantum_eval_v1.json`: 자체 작성 기준 질문 113개, 15개 분야
- 답변 가능 98개 / 답변 불가 15개
- `training_use: false`, 각 행 `split: held_out_eval_v1`
- 현재 `training_extra.json`의 228개 질문과 중복 0건 확인
- 데이터 구조 및 중복 검증: `validate_quantum_eval.py`

평가 데이터는 SFT, 사전학습, 토크나이저 병합, 후보 학습에 절대 사용하지 않습니다.

## 평가 실행
모델의 예측 결과를 JSONL로 준비합니다. 한 줄에 한 질문입니다.

```json
{"id":"QE-001","answer":"...","abstained":false,"citations":["source-id"]}
```

실행:
```bash
python evaluation/validate_quantum_eval.py
python evaluation/run_eval.py --predictions /path/to/predictions.jsonl
python evaluation/run_eval.py --predictions /path/to/predictions.jsonl --retrieval /path/to/retrieval.jsonl --output evaluation/report.json
python -m unittest discover -s evaluation -p 'test_*.py' -v
```

검색 결과 JSONL:
```json
{"id":"QE-001","retrieved_ids":["doc-a"],"relevant_ids":["doc-a"]}
```

## 지표 해석
- 필수 사실 포함률은 어휘 기반 proxy이며 의미적 정답률이 아닙니다.
- 무응답 재현율은 답변 불가 질문에서 보류했는지 측정합니다.
- 답변 가능 오거부율은 답변 가능한 질문을 불필요하게 거부한 비율입니다.
- 인용 존재율은 출처의 정확성·라이선스·내용 일치를 증명하지 않습니다.
- 검색 hit-rate/recall은 독립적으로 검수된 관련 문서 ID 정답표가 있어야 의미가 있습니다.
- 최종 사실 정확도, 문맥 적절성, 인용 품질은 사람 검토가 필요합니다.

이 평가기는 모델 학습, 승격, 배포를 수행하지 않습니다.


## 실제 Quantum 추론 서버 실행

먼저 HTTPS inference endpoint에 대해 3개 질문만 스모크 테스트한 뒤 전체 평가를 실행합니다. 토큰은 코드나 저장소에 기록하지 말고 환경 변수로 전달합니다.

```bash
export QUANTUM_API_TOKEN="(필요한 경우에만 설정)"
python evaluation/run_live_eval.py --limit 3 --out-dir evaluation/results-smoke
```

스모크 테스트 결과를 확인한 뒤 전체 113개를 실행합니다.

```bash
python evaluation/run_live_eval.py --out-dir evaluation/results
```

다른 Quantum endpoint를 검사하려면 `--endpoint https://호스트` 또는 `QUANTUM_EVAL_ENDPOINT`를 사용합니다. `--dry-run`은 네트워크 요청 없이 데이터셋과 인자만 확인합니다.

실행기는 각 요청의 모델명, 응답 모드, HTTP 상태, 지연 시간과 원본 JSON 응답을 저장합니다. 연결 실패나 HTTP 오류가 있으면 해당 질문의 오류를 기록하고 비정상 종료합니다. 무응답 감지는 문구 기반 휴리스틱이므로 반드시 사람이 원문 응답을 확인한 후 점수를 확정해야 합니다.

기본 endpoint의 health URL이 확인되더라도 실제 모델 로딩 및 체크포인트 상태를 보장하지 않습니다. 실행 전에 `GET /health`에서 `model_loaded`가 true인지 확인하고, false라면 생성 평가를 진행하지 마세요.
