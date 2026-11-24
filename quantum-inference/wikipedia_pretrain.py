"""Train a Wikipedia-derived causal-LM Candidate checkpoint for JUCKEX Quantum.

Input is a private JSONL export; no corpus or weights are committed to Git.
The checkpoint layout is compatible with quantum-inference/app.py.
This tool never deploys or changes Production.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from bpe_tokenizer import BPETokenizer
from tokenizer_diagnostics import summarize_tokenizer_usage

BLOCK = 256
LICENSE = "CC BY-SA 4.0"
WIKI_PREFIX = "https://ko.wikipedia.org/wiki/"


class BPECandidate(nn.Module):
    def __init__(self, vocab_size, width=64, heads=4, layers=2):
        super().__init__()
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(BLOCK, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 4,
            dropout=0.1, batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocab_size)
        self.register_buffer(
            "causal", torch.triu(torch.ones(BLOCK, BLOCK, dtype=torch.bool), diagonal=1)
        )

    def forward(self, ids):
        _, length = ids.shape
        x = self.token(ids) + self.position(torch.arange(length, device=ids.device))
        x = self.encoder(x, mask=self.causal[:length, :length])
        return self.output(self.norm(x))


def load_corpus(path, allow_unverified):
    docs, seen = [], set()
    rejected = {"invalid_json": 0, "invalid_metadata": 0, "duplicate": 0,
                "excluded": 0, "unverified_not_allowed": 0, "unsupported_status": 0}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{line_no}행 JSON 오류: {exc}") from exc
            if not isinstance(row, dict):
                rejected["invalid_metadata"] += 1
                continue
            string_fields = (
                "status", "reviewer_note", "page_id", "title", "url",
                "revision_id", "content_hash", "attribution", "text",
            )
            if any(not isinstance(row.get(field), str) for field in string_fields):
                rejected["invalid_metadata"] += 1
                continue
            status = row["status"].strip()
            note = row["reviewer_note"].strip()
            if status == "excluded":
                rejected["excluded"] += 1
                continue
            if status not in ("approved", "pending_review"):
                rejected["unsupported_status"] += 1
                continue
            review = row.get("provenance_review")
            reviewed_at = row.get("reviewed_at")
            identity_verified = (
                status == "approved"
                and note.startswith("MANUAL_APPROVAL_V2: ")
                and isinstance(review, dict)
                and review.get("approved") is True
                and review.get("reviewer_identity_verified") is True
                and review.get("identity_provider") == "cloudflare_access"
                and isinstance(review.get("reviewer"), str)
                and bool(review["reviewer"].strip())
                and isinstance(review.get("reviewer_subject"), str)
                and bool(review["reviewer_subject"].strip())
                and isinstance(reviewed_at, str)
                and _valid_timezone_aware_timestamp(reviewed_at)
                and review.get("reviewed_at") == reviewed_at
            )
            if status == "approved" and not identity_verified:
                rejected["invalid_metadata"] += 1
                continue
            if status == "pending_review" and (
                note or reviewed_at is not None or review is not None
            ):
                rejected["invalid_metadata"] += 1
                continue
            manually_verified = identity_verified
            if not manually_verified and not allow_unverified:
                rejected["unverified_not_allowed"] += 1
                continue
            page_id = row["page_id"].strip()
            title = row["title"].strip()
            url = row["url"].strip()
            revision_id = row["revision_id"].strip()
            digest = row["content_hash"].strip().lower()
            attribution = row["attribution"].strip()
            text = row["text"].strip()
            valid = (
                bool(page_id) and bool(title) and url.startswith(WIKI_PREFIX)
                and bool(re.fullmatch(r"[0-9]+", revision_id))
                and row.get("license") == LICENSE and bool(attribution)
                and len(text) >= 200
                and bool(re.fullmatch(r"[a-f0-9]{64}", digest))
                and hashlib.sha256(text.encode("utf-8")).hexdigest() == digest
            )
            if not valid:
                rejected["invalid_metadata"] += 1
                continue
            key = (page_id, digest)
            if key in seen:
                rejected["duplicate"] += 1
                continue
            seen.add(key)
            docs.append({
                "page_id": page_id, "title": title, "url": url,
                "revision_id": revision_id, "content_hash": digest,
                "license": LICENSE, "attribution": attribution,
                "trust": "identity_verified_human_review" if manually_verified else "unverified_experimental",
                "reviewed_at": reviewed_at if manually_verified else None,
                "reviewer": review.get("reviewer") if manually_verified else None,
                "reviewer_subject": review.get("reviewer_subject") if manually_verified else None,
                "identity_provider": review.get("identity_provider") if manually_verified else None,
                "text": title + "\n" + text,
            })
    if len(docs) < 2:
        raise ValueError(
            f"학습/검증 분리를 위해 유효 문서가 최소 2개 필요합니다. "
            f"현재 {len(docs)}개, 제외 내역={rejected}"
        )
    if any(d["trust"] == "unverified_experimental" for d in docs) and not allow_unverified:
        raise ValueError("미검증 문서가 포함되어 있습니다. 별도 실험 승인 플래그가 필요합니다.")
    return docs, rejected


def split_documents(docs, validation_fraction=0.1):
    try:
        fraction = float(validation_fraction)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("validation_fraction은 0과 1 사이의 유한한 숫자여야 합니다.") from exc
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction은 0과 1 사이의 유한한 숫자여야 합니다.")
    if not isinstance(docs, (list, tuple)) or len(docs) < 2:
        raise ValueError("학습/검증 분리를 위해 문서가 최소 2개 필요합니다.")
    if any(
        not isinstance(doc, dict)
        or not isinstance(doc.get("page_id"), str)
        or not doc["page_id"].strip()
        for doc in docs
    ):
        raise ValueError("모든 문서에는 문자열 형식의 비어 있지 않은 page_id가 필요합니다.")
    groups = {}
    for doc in docs:
        groups.setdefault(doc["page_id"], []).append(doc)
    page_ids = sorted(
        groups,
        key=lambda page_id: hashlib.sha256(page_id.encode("utf-8")).hexdigest()
    )
    if len(page_ids) < 2:
        raise ValueError("학습/검증 분리를 위해 서로 다른 page_id가 최소 2개 필요합니다.")
    valid_count = max(1, min(len(page_ids) - 1, round(len(page_ids) * fraction)))
    train_ids, valid_ids = set(page_ids[:-valid_count]), set(page_ids[-valid_count:])
    train_docs = [doc for doc in docs if str(doc["page_id"]) in train_ids]
    valid_docs = [doc for doc in docs if str(doc["page_id"]) in valid_ids]
    return train_docs, valid_docs


def make_windows(documents, tokenizer):
    windows = []
    token_count = 0
    for doc in documents:
        ids = tokenizer.encode(doc["text"])
        token_count += len(ids)
        if len(ids) < 2:
            continue
        for start in range(0, len(ids) - 1, BLOCK):
            chunk = ids[start:start + BLOCK + 1]
            if len(chunk) >= 2:
                windows.append(chunk)
    return windows, token_count


def batch_windows(windows, batch_size, device, vocab_size=None):
    if not isinstance(windows, (list, tuple)) or not windows:
        raise ValueError("배치 생성을 위해 비어 있지 않은 윈도우 목록이 필요합니다.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size는 1 이상의 정수여야 합니다.")
    if vocab_size is not None and (
        isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 1
    ):
        raise ValueError("vocab_size는 1 이상의 정수여야 합니다.")
    for index, window in enumerate(windows):
        if not isinstance(window, (list, tuple)) or len(window) < 2:
            raise ValueError(f"윈도우 {index}에는 최소 2개 토큰이 필요합니다.")
        if any(isinstance(token, bool) or not isinstance(token, int) or token < 0
               for token in window):
            raise ValueError(f"윈도우 {index}에 잘못된 토큰 ID가 있습니다.")
        if len(window) > BLOCK + 1:
            raise ValueError(f"윈도우 {index}가 BLOCK 크기를 초과합니다.")
        if vocab_size is not None and any(token >= vocab_size for token in window):
            raise ValueError(f"윈도우 {index}에 vocab_size 범위를 벗어난 토큰 ID가 있습니다.")
    selected = [random.choice(windows) for _ in range(batch_size)]
    x = torch.zeros((batch_size, BLOCK), dtype=torch.long)
    y = torch.full((batch_size, BLOCK), -100, dtype=torch.long)
    for index, chunk in enumerate(selected):
        inp, target = chunk[:-1], chunk[1:]
        x[index, :len(inp)] = torch.tensor(inp, dtype=torch.long)
        y[index, :len(target)] = torch.tensor(target, dtype=torch.long)
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, windows, batch_size, device, batches=20, vocab_size=None):
    model.eval()
    values = []
    for _ in range(min(batches, max(1, len(windows)))):
        x, y = batch_windows(windows, batch_size, device, vocab_size=vocab_size)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100
        )
        if torch.isfinite(loss):
            values.append(float(loss.item()))
    return sum(values) / len(values) if values else float("inf")


def atomic_torch_save(payload, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp",
                                     dir=str(destination.parent))
    os.close(fd)
    try:
        torch.save(payload, temp_name)
        os.replace(temp_name, destination)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def main():
    parser = argparse.ArgumentParser(description="Quantum Candidate Wikipedia causal-LM pretraining")
    parser.add_argument("--corpus", required=True, help="비공개 JSONL 문서 내보내기 파일")
    parser.add_argument("--output", default="candidate_wikipedia_pretrain.pt")
    parser.add_argument("--report", default="candidate_wikipedia_pretrain_report.json")
    parser.add_argument("--allow-unverified-experimental", action="store_true",
                        help="미검증 문서를 사실 검증 자료가 아닌 실험용 언어 코퍼스로 허용")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261122)
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 5000:
        parser.error("--steps는 1~5000 범위여야 합니다.")
    if args.batch_size < 1 or args.batch_size > 16:
        parser.error("--batch-size는 1~16 범위여야 합니다.")
    if args.width < 8 or args.heads < 1 or args.width % args.heads:
        parser.error("width는 8 이상이며 heads로 나누어져야 합니다.")
    if args.layers < 1 or args.layers > 8:
        parser.error("--layers는 1~8 범위여야 합니다.")

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    docs, rejected = load_corpus(args.corpus, args.allow_unverified_experimental)
    train_docs, valid_docs = split_documents(docs)
    tokenizer = BPETokenizer.train(
        [doc["text"] for doc in train_docs], vocab_size=args.vocab_size,
        min_pair_frequency=2,
        extra_alphabet_texts=[doc["text"] for doc in docs],
    )
    tokenizer_usage = {
        "training": summarize_tokenizer_usage(
            tokenizer, [doc["text"] for doc in train_docs], label="training"
        ),
        "validation": summarize_tokenizer_usage(
            tokenizer, [doc["text"] for doc in valid_docs], label="validation"
        ),
    }
    train_windows, train_tokens = make_windows(train_docs, tokenizer)
    valid_windows, valid_tokens = make_windows(valid_docs, tokenizer)
    if not train_windows or not valid_windows:
        raise RuntimeError("학습 또는 검증 토큰 윈도우가 비어 있습니다.")
    for doc in docs:
        if tokenizer.decode(tokenizer.encode(doc["text"])) != doc["text"]:
            raise RuntimeError(f"BPE 왕복 변환 실패: page_id={doc['page_id']}")

    device = torch.device("cpu")
    model = BPECandidate(len(tokenizer.vocab), args.width, args.heads, args.layers).to(device)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    best_loss = float("inf")
    best_state = None
    history = []
    model.train()
    for step in range(1, args.steps + 1):
        x, y = batch_windows(train_windows, args.batch_size, device, vocab_size=len(tokenizer.vocab))
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"학습 손실이 유효하지 않습니다: step={step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == args.steps:
            valid_loss = evaluate(model, valid_windows, args.batch_size, device, vocab_size=len(tokenizer.vocab))
            history.append({"step": step, "train_loss": float(loss.item()),
                            "validation_loss": valid_loss})
            if math.isfinite(valid_loss) and valid_loss < best_loss:
                best_loss = valid_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.train()

    if best_state is None:
        raise RuntimeError("검증 가능한 Candidate 체크포인트를 생성하지 못했습니다.")
    model.load_state_dict(best_state)
    trust_counts = {
        label: sum(1 for doc in docs if doc["trust"] == label)
        for label in ("identity_verified_human_review", "unverified_experimental")
    }
    manifest = [{
        "page_id": d["page_id"], "title": d["title"], "url": d["url"],
        "revision_id": d["revision_id"], "content_hash": d["content_hash"],
        "license": d["license"], "attribution": d["attribution"], "trust": d["trust"],
        "reviewed_at": d.get("reviewed_at"), "reviewer": d.get("reviewer"),
        "reviewer_subject": d.get("reviewer_subject"),
        "identity_provider": d.get("identity_provider"),
    } for d in docs]
    metadata = {
        "objective": "causal_next_token_prediction",
        "dataset": "private_wikipedia_jsonl_export",
        "document_count": len(docs), "train_document_count": len(train_docs),
        "validation_document_count": len(valid_docs),
        "train_token_count": train_tokens, "validation_token_count": valid_tokens,
        "tokenizer_usage": tokenizer_usage,
        "trust_counts": trust_counts, "rejected_rows": rejected,
        "reviewer_identity_provider": "cloudflare_access",
        "reviewer_identity_verified_document_count": trust_counts["identity_verified_human_review"],
        "license": LICENSE, "attribution_manifest": manifest,
        "unverified_corpus_is_fact_checked": False,
        "production_model_touched": False,
    }
    payload = {
        "state_dict": model.state_dict(),
        "tokenizer": tokenizer.to_dict(),
        "architecture": {"width": args.width, "heads": args.heads,
                         "layers": args.layers, "block": BLOCK},
        "steps": args.steps,
        "training_metadata": metadata,
    }
    atomic_torch_save(payload, args.output)
    report = {
        "model": "quantum-bpe-candidate-wikipedia",
        "parameters": parameter_count,
        "steps_requested": args.steps,
        "best_validation_loss": best_loss,
        "history": history,
        "tokenizer_usage": tokenizer_usage,
        "dataset": metadata,
        "checkpoint_path": str(Path(args.output)),
        "checkpoint_bytes": Path(args.output).stat().st_size,
        "checkpoint_load_compatible_with": "quantum-inference/app.py BPECandidate",
        "promotion_eligible": False,
        "human_quality_review_required": True,
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "ok": True, "parameters": parameter_count, "documents": len(docs),
        "train_documents": len(train_docs), "validation_documents": len(valid_docs),
        "best_validation_loss": best_loss, "checkpoint": str(Path(args.output)),
        "tokenizer_training_reduction": tokenizer_usage["training"]["token_reduction_ratio_vs_codepoints"],
        "tokenizer_validation_reduction": tokenizer_usage["validation"]["token_reduction_ratio_vs_codepoints"],
        "tokenizer_unknown_tokens": (
            tokenizer_usage["training"]["unknown_tokens"]
            + tokenizer_usage["validation"]["unknown_tokens"]
        ),
        "production_model_touched": False, "promotion_eligible": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
