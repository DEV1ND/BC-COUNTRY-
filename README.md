# JUCKEX / Quantum — 저장소 안내

이 저장소에는 기존 JUCKEX Production 구현과 분리된 Quantum Candidate 실험 코드가 함께 존재합니다. 두 실행 경로는 서로 다른 브랜치·서비스를 사용하며, Candidate 작업은 Production 모델을 수정하지 않습니다.

## 운영 환경 보호

- 기존 Production 서비스 및 Worker는 별도 운영 경로로 유지합니다.
- Candidate 개발 브랜치: `quantum-candidate-security-hardening`
- Candidate Render 서비스: `juckex-quantum-candidate`
- Candidate Render 자동 배포는 꺼져 있습니다. 코드 커밋만으로 서비스가 교체되지 않습니다.
- Candidate 검증과 사용자의 명시적 승인 전에는 Production 승격을 하지 않습니다.

## Quantum Candidate 구조

- `quantum-inference/`: BPE 토크나이저, Transformer 추론 API, 체크포인트 로더, 위키백과 지식 검색
- `quantum-inference/wikipedia_pretrain.py`: 별도 Candidate 사전학습 실험
- `quantum-inference/export_wikipedia_corpus.py`: 승인 등급을 구분한 읽기 전용 코퍼스 내보내기
- `quantum-inference/evaluation/benchmark_v1.json`: 학습에 사용하지 않는 15개 영역 평가 벤치마크
- `quantum-lab-worker/`: Candidate 실험용 Worker 코드

## 체크포인트 주의

체크포인트 바이너리는 공개 저장소에 넣지 않습니다. Candidate Render 서비스에 실제 체크포인트가 연결되어 있는지 `GET /health`의 `model_loaded`, `model_parameters`, `checkpoint_source`로 확인해야 합니다. 체크포인트가 없으면 추론 API는 HTTP 503을 반환해야 하며, 이를 정상 생성으로 보고하지 않습니다.

저장소의 과거 학습 보고서는 학습 실험 결과를 기록한 자료입니다. 보고서가 존재한다는 사실만으로 현재 Render 인스턴스에 가중치가 로드되어 있다는 뜻은 아닙니다.

## 테스트

```bash
cd quantum-inference
python -m unittest discover -s tests -v
python -m unittest -v test_auth_policy.py test_wikipedia_pretrain.py
```

GitHub Actions의 `Quantum Candidate encrypted training`은 먼저 Candidate 안전성 사전 검사를 수행합니다. 실제 Candidate 학습은 `dry_run=false`와 `confirm_candidate_training=true` 두 입력이 모두 명시된 경우에만 별도 작업으로 생성됩니다.

## 학습 데이터 provenance 및 신원 게이트

- `training_provenance.py`는 출처·라이선스·검토 메타데이터 완전성과 검토자 신원 검증을 별도로 계산합니다.
- `reviewer` 문자열, `reviewer_identity_verified` 데이터 필드, 승인 체크박스만으로 신원 인증을 통과할 수 없습니다.
- 기본 Candidate trainer는 신뢰된 암호학적 identity verifier를 연결하지 않으므로, 메타데이터가 완성되어도 학습 적격 판정은 계속 false입니다.
- `apply_training_provenance_review.py`는 검토 메타데이터가 입력된 별도 사본만 만들며, 학습 승인을 부여하지 않습니다.
- 실제 학습을 열려면 신뢰된 verifier 구현·키 관리·서명 검증 테스트·사람 검토를 별도로 완료해야 합니다.

## 위키백과 라이선스

위키백과 자료를 사용할 때 CC BY-SA 4.0의 저작자 표시 및 적용되는 동일조건변경허락 조건을 보존해야 합니다. 자동 검토 표시는 사람의 사실 검증이나 수동 승인을 의미하지 않습니다.

## 중요

README나 평가 보고서의 설명만으로 실제 배포·학습·추론 완료를 주장하지 않습니다. 작업 결과는 GitHub Actions 로그, Render 배포 기록, 실제 `/health` 응답 및 독립 평가 결과로 확인합니다.
