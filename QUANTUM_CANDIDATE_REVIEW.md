# Quantum Candidate 생성 품질 사람 검토

이 문서는 Candidate 체크포인트의 생성 답변을 검토하는 절차입니다. 운영 모델 승격을 자동으로 수행하지 않습니다.

## 검토 파일 만들기

10M 실험 GitHub Actions가 완료되면 `juckex-quantum-10m-smoke` 아티팩트에서 다음 파일을 받습니다.

- `quantum_decoding_sweep.json`: 디코딩 전략별 생성 결과와 참고용 표면 지표
- `quantum_human_review_template.json`: 사람이 점수를 입력할 검토 양식
- `bpe_candidate_report.json`: 학습·검증 메타데이터
- `bpe_candidate.json`: 후보 토크나이저 설정

## 검토 기준

각 생성 답변에 대해 1~5점으로 평가합니다.

| 항목 | 확인할 내용 |
|---|---|
| factuality | 사실과 기준 답변이 맞는가 |
| relevance | 질문에 직접 답하는가 |
| completeness | 핵심 내용이 빠지지 않았는가 |
| korean_naturalness | 한국어 표현이 자연스러운가 |

- 1: 매우 나쁨
- 2: 나쁨
- 3: 보통
- 4: 좋음
- 5: 매우 좋음

점수는 자동 추정하지 않습니다. 실제 생성 답변을 읽은 검토자가 직접 입력해야 합니다.

## 품질 게이트

`quantum_human_review.py --review-input <검토파일>` 명령으로 결과를 집계할 수 있습니다.

통과 조건:
- 모든 답변의 네 항목 점수가 입력되어야 합니다.
- 네 항목 각각의 평균이 4.0 이상이어야 합니다.
- 사실성 점수가 2 이하인 답변이 없어야 합니다.

이 조건은 후보 모델의 추가 검토를 위한 기준이며, 모델의 보편적 정확도를 증명하지 않습니다. 독립 평가 문항 20개는 제한된 표본입니다.

## 운영 보호

- 검토 통과만으로 Production에 자동 배포하지 않습니다.
- 사용자 승인과 별도 배포 검증이 필요합니다.
- 기존 Production 체크포인트와 Render 서비스를 변경하지 않습니다.
- 문자 bigram F1, BPC, 반복 비율은 참고 지표일 뿐 의미적 정답률이 아닙니다.

## 학습 데이터 출처 메타데이터 게이트

현재 저장소의 승인 표시 데이터 228개는 출처 메타데이터가 완전한 항목이 0개로 확인되었습니다. 기존 `source: "curated-v1"` 표시는 출처 종류·원본·라이선스·사용 권한을 검증하기에 충분하지 않습니다.

각 데이터 항목에 실제 근거를 확인한 뒤 다음 구조를 기록해야 합니다.

```json
{
  "source": {
    "kind": "실제 출처 유형",
    "reference": "확인 가능한 원본 또는 내부 작성 기록",
    "license": "확인된 라이선스",
    "attribution": "필요한 저작자 표시",
    "permission_basis": "사용 권한을 확인한 근거"
  },
  "provenance_review": {
    "approved": false,
    "reviewer": "",
    "reviewed_at": ""
  }
}
```

위 값은 형식 예시일 뿐입니다. 모르는 출처·라이선스·권한을 추측해 채우면 안 됩니다. 실제 검토자가 근거를 확인한 뒤 검토 상태를 갱신해야 합니다. 출처가 불명확한 항목은 후보 학습 데이터에서 제외하거나 보류합니다.

실행:
`python training_provenance.py --input training_extra.json --output training_provenance_audit.json`

학습 스크립트는 동일한 검사를 자체 수행하고, 검토가 끝나지 않은 데이터가 있으면 후보 학습을 중단합니다. 자동 감사 통과는 출처 주장의 진실성이나 라이선스 적법성을 보증하지 않습니다.


## 사람 검토 템플릿 적용 절차

`export_training_provenance_template.py`는 원본 Q&A를 바꾸지 않는 빈 검토 양식을 생성합니다. 사람이 실제 출처를 확인해 모든 필드를 작성하고 명시적으로 승인한 뒤에만 다음 가져오기 도구를 사용합니다.

```bash
python export_training_provenance_template.py --input training_extra.json --output training_provenance_review_template.json
```

검토 템플릿에서 각 항목의 다음 값을 실제 증거에 따라 작성합니다.

- `source.kind`: 사용자 직접 작성, 공개 데이터 등 실제 출처 유형
- `source.reference`: 원본 URL 또는 추적 가능한 내부 기록
- `source.license`: 확인된 라이선스
- `source.attribution`: 필요한 저작자·출처 표시
- `source.permission_basis`: 학습·재배포 등 목적에 대한 사용 권한 근거
- `provenance_review`: 검토자가 확인한 승인 여부, 검토자, ISO 8601 검토 시각

출처나 권한을 확인할 수 없는 항목은 승인하지 않습니다. 해당 항목을 학습에서 제외하려면 별도 Candidate 데이터 사본에서만 `approved`를 `false`로 변경한 뒤, 그 사본을 기준으로 검토 템플릿을 다시 생성합니다. 원본 데이터 파일은 직접 덮어쓰지 않습니다.

작성한 템플릿은 원본과 질문·답변 해시·위치를 대조해 적용합니다.

```bash
python apply_training_provenance_review.py \\
  --input training_extra_candidate.json \\
  --template training_provenance_review_template.json \\
  --output training_extra_reviewed_candidate.json
python training_provenance.py \\
  --input training_extra_reviewed_candidate.json \\
  --output training_provenance_audit.json --strict
```

가져오기 도구는 새 출력 파일만 생성하며 기존 파일을 덮어쓰지 않습니다. 누락·중복·변경된 질문/답변·빈 메타데이터·미승인 검토·잘못된 날짜가 있으면 실패합니다. 구조 검증 통과는 출처 주장의 진실성을 독립적으로 증명하지 않습니다. 따라서 실제 증거 검토 책임은 사람에게 있으며, 이 절차만으로 Production 승격이나 배포는 수행되지 않습니다.
