# juckex — 자체 Transformer 자동 학습 서버 v2

기존 `juckex_weights.npz`와 `vocab.json`을 그대로 사용하며, 기존 4층 Transformer를 PyTorch로 복원해 이어서 미세조정합니다. 사전학습 모델이나 외부 AI 답변 API는 사용하지 않습니다.

## 자동 학습 흐름

1. juckex 채팅에서 모르는 질문은 학습 후보로 저장됩니다.
2. 관리자가 후보에 정확한 답변을 승인하거나 검증된 지식을 학습 데이터로 등록합니다.
3. 승인된 Q&A는 Cloudflare D1의 `training_pairs`에 저장됩니다.
4. Cloudflare Worker Cron이 현재 설정된 시간 간격(운영 설정 기준 1시간)에 새 데이터를 확인합니다.
5. 마지막 성공 시점보다 승인 데이터가 5개 이상 늘면 Render의 `/train/start`를 호출합니다.
6. Render가 D1 데이터 전체를 페이지 단위로 가져온 뒤, 무작위 배치로 실제 Transformer 가중치를 최대 80 step 미세조정합니다. 기본 80 step에서는 최대 640개 시퀀스가 샘플링되며, 9만 건 전체를 한 번에 모두 순회 학습하는 것은 아닙니다. `dataset_pairs`는 가져온 데이터 수, `examples_seen`은 실제 학습 배치에 투입된 시퀀스 수입니다.
7. 고정된 검증 배치의 손실이 학습 전보다 15% 넘게 나빠지면 새 가중치를 거부합니다.
8. 통과한 가중치는 Cloudflare KV에 저장되고, 이전 가중치는 백업됩니다. Render가 재시작되어도 KV에서 모델을 복원합니다.

**모든 대화가 무조건 자동 학습되는 방식은 아닙니다.** 틀린 답변이 모델에 들어가는 것을 막기 위해 승인된 데이터만 학습합니다. 새 데이터가 5개 미만이면 다음 Cron 실행까지 대기합니다.

## 배포

이 패키지는 기존 Render 서비스 `juckex-inference`에 연결된 Git 저장소에 반영한 뒤 재배포하는 용도입니다.

1. 압축을 풀고 `app.py`, `requirements.txt`, `render.yaml`, `vocab.json`, `juckex_weights.npz`를 기존 Render 저장소의 루트에 덮어씁니다.
2. Render에서 기존 `juckex-inference` 서비스를 재배포합니다.
3. Render의 Environment에 다음 항목이 있는지 확인합니다.
   - `JUCKEX_WORKER_URL` = `https://juckex-ai.bcsite.workers.dev`
   - `JUCKEX_SYNC_TOKEN` = Cloudflare Worker에 등록된 `JUCKEX_SYNC_TOKEN` Secret과 **완전히 동일한 값**
4. 기존 서비스에 같은 환경 변수가 이미 있다면 그대로 유지하면 됩니다. 토큰은 공개 Git 저장소에 올리지 마세요.

`render.yaml`은 Free Web Service 설정을 유지합니다. Render 무료 인스턴스는 유휴 상태에서 잠들 수 있으므로, 15분 Cron 요청이 서비스를 깨운 뒤 학습을 요청할 수 있습니다. 학습이 시작되기까지 지연이 생길 수 있습니다.

## 확인 주소

- 서비스: https://juckex-inference.onrender.com/
- 상태: https://juckex-inference.onrender.com/health
- 모델/학습 상태: https://juckex-inference.onrender.com/model/status

`/health` 응답의 `remote_storage_configured`가 `true`여야 자동 학습 결과를 Cloudflare KV에 보존할 수 있습니다.

## 주요 API

- `GET /health` — 서비스 및 원격 저장소 설정
- `POST /generate` — 기존 Transformer 추론
- `GET /model/status` — 실제 Transformer 학습 상태
- `POST /train/start` — 인증된 자동 학습 요청
- `POST /model/rollback` — 인증된 이전 모델 복구
- `GET /` — 서비스 준비 상태

학습·복구 API는 `X-Juckex-Token` 헤더를 요구합니다.

## 보호 및 제한

- 관리자 승인 및 검증된 데이터만 D1 학습 저장소에 들어가도록 운영합니다.
- 학습 전후 동일한 검증 배치를 사용해 비교합니다.
- 검증 손실이 15% 넘게 악화되면 기존 모델을 유지합니다.
- 새 가중치를 KV에 저장하지 못하면 메모리의 모델도 교체하지 않습니다.
- Render의 임시 디스크에만 저장된 모델은 성공 처리하지 않습니다.
- 기존 토크나이저는 447개 문자로 고정되어 있습니다. 어휘에 없는 문자는 제대로 학습되지 않습니다.
- 이 모델은 약 92만 파라미터의 소형 문자 단위 Transformer입니다. 자동 학습이 된다고 해서 범용 AI 수준의 지식·정확도가 보장되는 것은 아닙니다.

## 원본 보존

배포 패키지의 `juckex_weights.npz`와 `vocab.json`은 기존 체크포인트를 유지합니다. 첫 학습은 이 가중치에서 이어지며, Cloudflare KV에는 현재 모델과 직전 모델을 별도로 저장합니다.


## Quantum 무료 Colab 학습 실험

- 노트북: [quantum-inference/Quantum_Free_Colab_Training.ipynb](https://github.com/DEV1ND/BC-COUNTRY-/blob/quantum-v2-development/quantum-inference/Quantum_Free_Colab_Training.ipynb)
- 이 노트북은 무료 Colab 세션에서 승인된 한국어 위키백과 문서로 독립적인 후보 체크포인트를 학습합니다.
- Cloudflare의 격리된 Wikipedia Worker에 인증된 `GET /export-approved` 경로가 배포되어 있어야 합니다. 이 경로는 승인된 본문을 최대 10개씩 페이지 단위로 내보냅니다.
- 후보 모델은 파일로만 내려받으며 운영 JUCKEX 또는 운영 Quantum 모델을 자동 교체하지 않습니다.
- 무료 Colab GPU는 세션·할당량이 제한되고 학습 결과는 세션 종료 후 사라질 수 있으므로 결과 체크포인트를 직접 보관해야 합니다.
