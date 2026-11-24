# Quantum Candidate: 암호화된 수동 학습 실행

이 절차는 **Candidate 전용**입니다. GitHub Actions의 일회성 CPU 러너에서 작은 파일럿 학습을 실행하고, 체크포인트와 상세 평가 보고서를 AES-256-GCM으로 암호화한 뒤 7일 보관 아티팩트로 전달합니다. 원문 코퍼스와 평문 체크포인트는 저장소에 커밋하거나 아티팩트에 업로드하지 않습니다.

- Production 모델/Render 서비스는 변경하지 않습니다.
- Worker/D1/KV를 수정하지 않습니다. Worker export는 읽기 전용입니다.
- 자동 학습이 아닙니다. 저장소 관리자가 직접 workflow_dispatch를 실행해야 합니다.
- 첫 실행은 25개 이하 문서, 50 step 권장입니다.
- Validation loss는 사실 정확도나 모델 승격 기준이 아닙니다.
- 이 작업은 Candidate 학습 실험이며 Render 배포/체크포인트 연결을 자동 수행하지 않습니다.

## 1. GitHub Actions secrets 준비

저장소 관리자가 GitHub 저장소의 **Settings → Secrets and variables → Actions → New repository secret**에서 아래 두 항목을 추가해야 합니다.

1. `JUCKEX_WIKI_ADMIN_TOKEN`: Cloudflare Worker의 기존 `ADMIN_TOKEN`. Cloudflare 대시보드에서 직접 복사해 GitHub Secret에 입력합니다. ChatGPT 대화나 코드에 붙여넣지 마세요.
2. `QUANTUM_CHECKPOINT_ENCRYPTION_KEY`: 32바이트(64자리 16진수) 임의 키. 로컬 Python에서 아래 명령으로 생성하고, 화면에 나온 값은 비밀 관리자만 저장합니다.

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

암호화 키는 체크포인트 복호화에 반드시 필요합니다. 분실하면 암호화된 파일을 복구할 수 없습니다. 저장소 코드, 이슈, 커밋, 로그, 대화창에 키를 기록하지 마세요. 키를 잃어버리면 새 키로 재실행해야 합니다.

## 2. 수동 실행

GitHub → Actions → **Quantum Candidate encrypted training** → Run workflow에서 실행합니다.

- `manual`: 사람이 직접 승인한 문서만 사용합니다. 최소 2개 문서가 있어야 합니다.
- `unverified`: `allow_unverified_experimental=true` 확인을 별도로 선택해야 합니다. 이는 사실 검증이 아닌 실험용 언어 코퍼스입니다. 생성 답변을 검증된 사실로 취급하면 안 됩니다.
- 문서 상한은 10/25/50, 스텝은 25/50/100 중에서 선택합니다.

초기 실험은 25개 문서와 50 step으로 제한합니다. 전체 500여 문서를 한 번에 학습하는 설정이 아닙니다.

## 3. 암호화된 아티팩트 복호화

성공한 실행의 Summary에 체크포인트 파라미터 수, 학습 문서 수, validation loss가 표시됩니다. Actions 실행 화면의 Artifacts에서 `quantum-candidate-encrypted-<run id>`를 내려받습니다. 아티팩트에는 암호화된 체크포인트/상세 보고서와 민감한 본문을 포함하지 않는 요약만 들어 있습니다.

아래 스크립트를 로컬에 `decrypt_quantum_candidate.py`로 저장합니다. `QUANTUM_CHECKPOINT_ENCRYPTION_KEY`를 로컬 환경 변수로 설정한 뒤 실행합니다.

```python
import os
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

key = bytes.fromhex(os.environ["QUANTUM_CHECKPOINT_ENCRYPTION_KEY"])
if len(key) != 32:
    raise SystemExit("Encryption key must be 32 bytes")

def decrypt(source, destination):
    source = Path(source)
    data = source.read_bytes()
    if data[:6] != b"JQENC1" or len(data) < 34:
        raise ValueError(f"Invalid encrypted file: {source.name}")
    nonce = data[6:18]
    aad = source.name.encode("utf-8")
    plaintext = AESGCM(key).decrypt(nonce, data[18:], aad)
    target = Path(destination)
    target.write_bytes(plaintext)
    os.chmod(target, 0o600)
    print(f"Decrypted: {target}")

decrypt("quantum-candidate.pt.enc", "candidate_wikipedia_pretrain.pt")
decrypt("quantum-candidate-report.json.enc", "candidate_wikipedia_pretrain_report.json")
```

파일 이름을 바꾸기 전에 복호화하세요. 암호화는 파일명도 인증 데이터(AAD)로 사용합니다. 복호화한 체크포인트는 비공개로 보관하고 공개 저장소/아티팩트에 올리지 마세요.

## 4. 다음 단계와 제한

현재 Candidate Render 서비스에는 영구 체크포인트 저장 공간이 확인되지 않았습니다. 따라서 복호화된 파일을 Render에 자동 배포하거나 Production에 연결하지 않습니다. 별도의 안전한 영구 저장소가 준비되고 체크포인트 로딩/추론 평가가 통과한 뒤에도, Candidate 검증과 사용자 승인 전에는 Production 승격을 하지 않습니다.

위키백과의 CC BY-SA 4.0 출처·저작자 표시 및 적용되는 동일조건변경허락 의무를 보존해야 합니다. 암호화 아티팩트는 7일 뒤 만료되므로 필요한 파일은 관리자가 안전한 장소에 보관해야 합니다.


## 5. 토크나이저 및 검색 입력 검증 게이트

Candidate 토크나이저는 등록된 초기 문자와 BPE 병합 목록으로 어휘를 구성합니다.

- 학습/검증에 사용할 문자 범위는 `extra_alphabet_texts`로 명시적으로 제공하고, 이 범위에 포함된 Unicode 코드포인트는 원문 그대로 왕복 인코딩/디코딩되어야 합니다.
- 등록되지 않은 문자는 `<unk>`로 인코딩되며, 디코딩 시 대체 문자(`�`)로 표시됩니다. 이는 원문 복원이 보장되지 않는다는 뜻이며, 모델 품질 평가에서 별도로 다뤄야 합니다.
- Unicode 정규화(NFC/NFKC)를 토크나이저가 임의 적용하지 않습니다. 결합 문자, 자모, 이모지 조합, 줄바꿈 등은 원본 문자열 보존 테스트로 확인합니다.
- 직렬화 후 토크나이저를 다시 불러왔을 때 어휘 및 인코딩 결과가 동일해야 합니다.
- 검색 질의, 승인 Q&A, 검색 문서의 필드 타입은 명시적으로 검증하며, 잘못된 숫자/문자열 자동 변환을 허용하지 않습니다.
- 위키백과 학습 코퍼스도 상태, 검수 메모, 문서 ID, 제목, URL, 리비전, 해시, 저작자 표시, 본문을 문자열로 엄격히 요구합니다. 잘못된 타입은 문자열로 강제 변환하지 않고 해당 행을 제외합니다. 문서 분할 시 page_id 역시 문자열이어야 합니다.
- CI 통과는 코드 계약 검증일 뿐, 토크나이저의 한국어 언어 품질이나 Quantum 답변의 사실 정확도를 증명하지 않습니다.


## 6. 검색 인덱스 오류 격리

Candidate 추론 API의 문서 검색은 검색 결과와 검색 상태를 분리해 기록합니다.

- `ok`: FTS5 검색과 관련성 재정렬이 정상적으로 완료되어 문서가 반환됨
- `no_match`: 검색은 정상 완료됐지만 관련 문서가 없음
- `empty_query`: 의미 검색어가 없어 DB 조회를 수행하지 않음
- `fts_fallback`: FTS5를 사용할 수 없거나 결과가 없어 원본 `knowledge_docs` 테이블의 제한된 LIKE 검색을 사용함
- `index_invalid`: 인덱스 문서 필드의 타입이 잘못되어 문서를 신뢰하지 않고 제외함
- `search_unavailable`: SQLite 검색을 완료하지 못함

관리자 검색 진단 응답은 `retrieval_status`와 `used_fallback`을 반환합니다. 일반 모델 생성 응답도 `knowledge_retrieval` 메타데이터에 상태와 실제 전달된 출처 수를 기록합니다. 검색 오류가 발생해도 오류 문서를 근거로 사용하지 않으며, 체크포인트가 정상일 때는 검색 문맥 없이 Candidate 모델 생성 단계로 계속 진행할 수 있습니다.

이 상태 코드는 검색 시스템의 가용성 및 입력 무결성을 나타낼 뿐, 검색된 문서의 사실 정확도나 생성 답변의 의미적 정확도를 보증하지 않습니다. `index_invalid`는 데이터 손상을 숨기거나 자동 수정하지 않는다는 뜻이며, 원본 문서와 인덱스 상태를 별도로 조사해야 합니다.


## 7. 지식 문서 가져오기 검증

Candidate의 `/admin/knowledge/import`는 쓰기 전에 전체 요청 배치를 검증합니다.

- 문서 식별자, 제목, URL, 본문, 라이선스, 저작자 표시, 리비전 ID, SHA-256은 모두 문자열이어야 하며 숫자나 불리언을 문자열로 자동 변환하지 않습니다.
- URL/제목/저작자 표시 및 CC BY-SA 4.0 라이선스 정보가 일치해야 합니다.
- 리비전 ID는 숫자 문자열이어야 합니다.
- 본문 해시는 실제 UTF-8 본문으로 계산한 SHA-256과 정확히 일치해야 합니다.
- 본문 길이는 200~50,000자로 제한하며, 배치는 1~10개 문서로 제한합니다.
- 한 문서라도 검증에 실패하면 데이터베이스 쓰기를 시작하지 않습니다.
- 저장 도중 원본 문서 또는 FTS 인덱스 쓰기가 실패하면 전체 트랜잭션을 롤백합니다.

이 검증은 요청 데이터의 구조와 무결성을 확인합니다. 원문이 사실인지, 제출자가 실제 저작자 표시 의무를 충족했는지까지 자동 증명하지는 않습니다. 문서 수집 단계의 출처·라이선스 검토는 별도로 유지해야 합니다.
