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


## 5. Candidate Render의 재시작 복원 설정

Render Free에는 영구 디스크가 없으므로 Candidate 서버가 재시작할 때 암호화된 GitHub Release 체크포인트를 다시 받아야 합니다.

학습 Workflow가 생성한 릴리스의 파일 주소를 Candidate Render 서비스 환경 변수에 설정합니다.

- `QUANTUM_CHECKPOINT_URL`: `https://github.com/DEV1ND/BC-COUNTRY-/releases/download/quantum-candidate-RUN_ID/quantum-candidate.pt.enc` 형식의 실제 릴리스 파일 URL
- `QUANTUM_CHECKPOINT_ENCRYPTION_KEY`: GitHub Actions의 동일한 64자리 키. Render 대시보드에서 직접 설정하며 대화나 코드에 입력하지 않습니다.
- `QUANTUM_API_TOKEN`: 기존 Candidate 인증 토큰 유지

이 기능은 Candidate 서버가 시작할 때만 복원하며 기존 로컬 체크포인트를 덮어쓰지 않습니다. 복원 실패 시 서버는 체크포인트 미로드 상태로 남고, 모델이 작동한다고 표시하지 않습니다. `/health`에서 `model_loaded`, 실제 파라미터 수, `checkpoint_restore_status`를 확인합니다.

Production 서비스에는 이 환경 변수를 설정하지 않습니다.
