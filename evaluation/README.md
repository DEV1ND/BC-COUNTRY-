# QUANTUM EVALUATION v1

이 디렉터리는 JUCKEX Quantum의 독립 평가 전용 자료를 보관합니다. 어떤 파일도 사전학습, SFT, 토크나이저 병합 학습 또는 후보 모델의 gradient update에 사용하면 안 됩니다.

## 핵심 평가 데이터
- `quantum_eval_v1.json`: 113개 한국어 평가 질문. 15개 분야, 답변 가능 98개, 정보 부족 시 무응답해야 하는 15개를 포함합니다.
- `validate_quantum_eval.py`: 구조, 중복, 분야 수, 학습 데이터 및 기존 독립 평가 세트와의 질문 중복을 검사합니다.
- `quantum_eval_v1.jsonl` 및 `run_eval.py`: 기존 QEV1-001 형식의 예측 결과 평가기입니다. 문자 키워드 포함률은 의미 정확도가 아닌 참고 지표입니다.

## 데이터 검증
저장소 루트에서 실행합니다.

```bash
python evaluation/validate_quantum_eval.py
```

## 기존 예측 평가기
```bash
python evaluation/run_eval.py --predictions /path/to/predictions.jsonl
python evaluation/run_eval.py --predictions /path/to/predictions.jsonl --retrieval /path/to/retrieval.jsonl --output evaluation/report.json
```

예측 JSONL 행:
```json
{"id":"QEV1-001","answer":"...","abstained":false,"citations":["source-id"]}
```

검색 JSONL 행:
```json
{"id":"QEV1-001","retrieved_ids":["doc-a"],"relevant_ids":["doc-a"]}
```

## 평가 원칙
- 검색 정확도와 답변 품질을 별도로 보고합니다.
- 검색 결과가 존재한다고 정답으로 간주하지 않습니다.
- 답변 가능 질문은 사실 정확성, 필수 사실 포함, 관련성, 완결성, 인용 근거를 사람이 검토해야 합니다.
- 답변 불가능 질문은 정보 부족을 명확히 밝히고 사실을 지어내지 않아야 통과입니다.
- 키워드 포함률, BPC, 검색 hit rate는 의미적 정답률이 아닙니다.
- 평가 실행마다 모델/체크포인트 해시, API 버전, 디코딩 설정, 실행 시각, 원시 응답을 보존합니다.
- 모델 승격이나 운영 배포는 이 평가 도구에서 수행하지 않습니다.

## 출처와 라이선스
`quantum_eval_v1.json`의 질문과 기준 답변은 프로젝트 평가용으로 자체 작성했습니다. 위키백과 등 외부 자료의 문장을 복사한 데이터가 아닙니다. 향후 외부 자료를 추가할 때는 출처와 라이선스를 별도로 확인하고 기록해야 합니다.
