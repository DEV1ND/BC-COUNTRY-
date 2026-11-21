"""Local checkpoint smoke test; requires QUANTUM_CHECKPOINT_PATH."""
import os
os.environ.setdefault("QUANTUM_CHECKPOINT_PATH", "bpe_candidate_checkpoint.pt")
from app import model, tokenizer
import torch
if model is None:
    raise SystemExit("CHECKPOINT_NOT_LOADED")
for question in ["태양계에서 가장 큰 행성은?", "판 구조론이란 무엇인가요?", "오늘 날씨 어때?"]:
    ids = tokenizer.encode("질문:" + question + "\n답변:")
    generated = []
    with torch.inference_mode():
        for _ in range(64):
            x = torch.tensor((ids + generated)[-256:], dtype=torch.long).unsqueeze(0)
            logits = model(x)[0, -1]
            token = int(torch.argmax(logits))
            generated.append(token)
            decoded = tokenizer.decode(generated)
            if "\n\n" in decoded:
                break
    print({"question": question, "answer": tokenizer.decode(generated).strip()})
