# juckex inference

독립 학습한 작은 문자 단위 Transformer의 추론 서버입니다. 외부 AI API나 사전학습 모델을 호출하지 않습니다.

## 파일
- `app.py`: NumPy 기반 Transformer 추론 API
- `juckex_weights.npz`: 학습 체크포인트에서 추출한 모델 가중치
- `vocab.json`: 문자 사전 및 모델 설정
- `render.yaml`: Render 배포 설정

## 로컬 실행
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```
`POST /generate` 요청 예시:
```json
{"prompt":"질문: 인공지능이란?\n답변:", "max_new_tokens":160}
```

## 참고
현재 포함된 모델은 무작위 초기화부터 2,000 step 학습한 실험용 모델입니다. 실제 응답 API는 작동하지만 답변의 정확성·일관성은 제한적이며, 더 많은 고품질 데이터와 학습이 필요합니다.
