# Quantum Candidate Immutable Checkpoint Contract

## 목적

후보 체크포인트를 운영 모델과 혼동하거나 같은 저장 객체에 덮어쓰는 일을 방지하기 위한 **저장 계약**입니다. 이 문서는 저장소 계약과 메타데이터 검증을 정의하며, 실제 Worker 저장 API를 구현하거나 배포하지 않습니다.

## 계약

- 역할은 항상 `candidate`이며 `production_eligible=false`입니다.
- 체크포인트 ID는 실제 바이너리의 SHA-256으로 결정합니다.
- 저장 키는 `candidate/checkpoints/sha256/<digest>.npz` 형태입니다.
- 아키텍처 ID, 파라미터 수, 학습 step, 데이터 fingerprint, provenance digest, validation loss, 생성 시각을 기록합니다.
- 생성 시각은 timezone이 명시된 UTC ISO-8601이어야 합니다.
- 동일한 바이트는 동일한 ID를 만들고, 바이트가 다르면 ID도 달라집니다.

## Worker 저장 계층에 필요한 동작

향후 Worker endpoint를 만들 때 다음을 모두 구현해야 합니다.

1. 전용 Candidate namespace 또는 전용 R2 bucket을 사용하고 Production 모델 키와 공유하지 않습니다.
2. `PUT`은 create-only 조건부 쓰기(이미 존재하면 409)를 사용합니다. 동일 키를 덮어쓰지 않습니다.
3. 업로드된 바이너리 SHA-256을 Worker가 직접 계산하고 manifest와 비교합니다.
4. D1에는 manifest와 검증 상태를 저장하되, D1 행만으로 바이너리 무결성을 증명했다고 간주하지 않습니다.
5. 조회는 checkpoint ID로만 수행하고, 삭제/수정은 별도 관리자 승인 절차 없이는 허용하지 않습니다.
6. Candidate API 인증은 기존 sync token을 재사용하더라도 Production 모델 쓰기 권한과 분리해야 합니다.
7. 운영 승격은 독립 평가 기록과 사람의 명시적 승인 후 별도 작업으로만 수행합니다.

## 현재 제한

- 저장소에서 기존 `/api/internal/checkpoint` 호출은 확인했지만, 그 endpoint를 구현한 Worker 소스 경로는 현재 확인되지 않았습니다.
- 따라서 이 계약은 기존 endpoint에 연결하지 않았습니다. 기존 endpoint는 현재 단일 객체를 재사용하는 것으로 문서화되어 있어 immutable 저장을 보장한다고 가정할 수 없습니다.
- 실제 Candidate 저장, Render 연결, Cloudflare D1/R2 변경, 학습 및 배포는 수행하지 않았습니다.

## 검증

`python -m unittest -v test_candidate_checkpoint_contract.py`

이 테스트는 manifest 계약만 검증합니다. Worker의 조건부 저장, 인증, 동시성, 실제 R2/D1 동작은 별도의 통합 테스트가 필요합니다.
