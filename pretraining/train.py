"""Isolated autoregressive pretraining for Quantum candidates.

Consumes only locally staged, provenance-reviewed plain-text corpus files.
Never imports the production app and never writes to production checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from bpe_tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
# Conservative default: licenses requiring share-alike or dataset-specific approval
# must be reviewed separately before being admitted to the training corpus.
ALLOWED_LICENSES = {"CC0-1.0", "CC-BY-4.0"}


def load_corpus(manifest_path: Path, min_chars: int = 200):
    records = []
    seen_hashes: dict[str, str] = {}
    for line_no, raw in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        required = ("path", "source_name", "source_url", "retrieved_at", "license_id", "license_url", "attribution", "split", "review_status")
        missing = [key for key in required if not str(row.get(key, "")).strip()]
        if missing:
            raise ValueError(f"manifest line {line_no}: missing {', '.join(missing)}")
        if row["review_status"] != "approved":
            raise ValueError(f"manifest line {line_no}: document is not manually approved")
        if row["split"] not in ("train", "validation"):
            raise ValueError(f"manifest line {line_no}: split must be train or validation")
        license_id = str(row["license_id"]).strip().upper()
        if license_id not in ALLOWED_LICENSES:
            raise ValueError(
                f"manifest line {line_no}: license {license_id!r} is not in the conservative allowlist "
                f"{sorted(ALLOWED_LICENSES)}; obtain a documented legal/use-policy review first"
            )
        path = (ROOT / str(row["path"])).resolve()
        if ROOT.resolve() not in path.parents:
            raise ValueError(f"manifest line {line_no}: corpus path escapes pretraining directory")
        if not path.is_file() or path.suffix.lower() not in (".txt", ".md"):
            raise ValueError(f"manifest line {line_no}: expected an existing .txt or .md file: {path}")
        text = path.read_text(encoding="utf-8").strip()
        if len(text) < min_chars:
            raise ValueError(f"manifest line {line_no}: text shorter than {min_chars} characters")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        prior = seen_hashes.get(digest)
        if prior and prior != row["split"]:
            raise ValueError("identical document content crosses train/validation split")
        if prior:
            raise ValueError(f"duplicate document content in manifest: {path.name}")
        seen_hashes[digest] = row["split"]
        records.append({
            "text": text, "split": row["split"], "source_name": row["source_name"],
            "source_url": row["source_url"], "retrieved_at": row["retrieved_at"],
            "license_id": row["license_id"], "license_url": row["license_url"],
            "attribution": row["attribution"],
            "category": row.get("category", "unspecified"),
            "reviewed_by": row.get("reviewed_by", "manifest-review"),
            "sha256": digest, "path": str(path.relative_to(ROOT)),
        })
    train = [r for r in records if r["split"] == "train"]
    valid = [r for r in records if r["split"] == "validation"]
    if not train or not valid:
        raise ValueError("manifest must contain both train and validation documents")
    return train, valid


def pack_tokens(texts, tokenizer):
    ids = []
    for text in texts:
        ids.extend(tokenizer.encode(text))
        ids.append(tokenizer.stoi.get("\n", 0))
    if len(ids) < 2:
        raise ValueError("not enough tokens to train")
    return torch.tensor(ids, dtype=torch.long)


class QuantumPretrainingModel(nn.Module):
    def __init__(self, vocab_size: int, width: int, heads: int, layers: int, block_size: int):
        super().__init__()
        self.block_size = block_size
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(block_size, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 4,
            dropout=0.1, batch_first=True, norm_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)
        self.register_buffer("causal_mask", torch.triu(
            torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1
        ))

    def forward(self, input_ids):
        length = input_ids.size(1)
        positions = torch.arange(length, device=input_ids.device)
        x = self.token(input_ids) + self.position(positions)
        x = self.transformer(x, mask=self.causal_mask[:length, :length])
        return self.lm_head(self.norm(x))


def sample_batch(tokens, batch_size, block_size, device):
    if len(tokens) <= block_size + 1:
        raise ValueError(f"corpus needs more than {block_size + 1} tokens; got {len(tokens)}")
    starts = torch.randint(0, len(tokens) - block_size - 1, (batch_size,))
    x = torch.stack([tokens[s:s + block_size] for s in starts])
    y = torch.stack([tokens[s + 1:s + block_size + 1] for s in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, tokens, batches, batch_size, block_size, device):
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = sample_batch(tokens, batch_size, block_size, device)
        logits = model(x)
        losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item())
    model.train()
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser(description="Train an isolated Quantum pretraining candidate")
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifest.jsonl")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "quantum_pretraining_candidate_10k.pt")
    args = parser.parse_args()
    if args.steps != 10_000:
        raise ValueError("official 10K run requires --steps 10000; use a separate smoke-test script for short tests")
    if args.width <= 0 or args.heads <= 0 or args.width % args.heads:
        raise ValueError("width must be positive and divisible by heads")
    if args.block_size < 8 or args.batch_size < 1 or args.eval_interval < 1 or args.eval_batches < 1:
        raise ValueError("invalid batch, block, or evaluation interval")
    output = args.output.resolve()
    artifacts_root = (ROOT / "artifacts").resolve()
    if artifacts_root not in output.parents or output.suffix.lower() != ".pt":
        raise ValueError("candidate output must be a .pt file inside pretraining/artifacts; production paths are forbidden")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite candidate artifact: {output}")
    args.manifest = args.manifest.resolve()
    if not args.manifest.is_file():
        raise FileNotFoundError(f"missing corpus manifest: {args.manifest}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_rows, valid_rows = load_corpus(args.manifest)
    # BPE merge statistics are learned from training documents only.
    train_texts = [r["text"] + "\\n" for r in train_rows]
    validation_texts = [r["text"] + "\\n" for r in valid_rows]
    tokenizer = BPETokenizer.train(
        train_texts, vocab_size=args.vocab_size,
        min_pair_frequency=3, extra_alphabet_texts=validation_texts,
    )
    train_tokens = pack_tokens([r["text"] for r in train_rows], tokenizer)
    valid_tokens = pack_tokens([r["text"] for r in valid_rows], tokenizer)
    model = QuantumPretrainingModel(
        len(tokenizer.vocab), args.width, args.heads, args.layers, args.block_size
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    best_val = math.inf
    output.parent.mkdir(parents=True, exist_ok=True)
    history = []

    for step in range(1, args.steps + 1):
        model.train()
        x, y = sample_batch(train_tokens, args.batch_size, args.block_size, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.steps:
            val_loss = evaluate(model, valid_tokens, args.eval_batches, args.batch_size, args.block_size, device)
            history.append({"step": step, "train_loss": round(loss.item(), 6), "validation_loss": round(val_loss, 6)})
            print(json.dumps(history[-1]), flush=True)
            if math.isfinite(val_loss) and val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "format": "juckex-quantum-pretraining-candidate-v1",
                    "model_state_dict": model.state_dict(),
                    "tokenizer": tokenizer.to_dict(),
                    "architecture": {
                        "vocab_size": len(tokenizer.vocab), "width": args.width,
                        "heads": args.heads, "layers": args.layers,
                        "block_size": args.block_size,
                    },
                    "training": {
                        "steps_completed": step, "requested_steps": args.steps,
                        "seed": args.seed, "best_validation_loss": best_val,
                        "history": history, "device": str(device),
                        "training_documents": len(train_rows), "validation_documents": len(valid_rows),
                        "training_tokens": len(train_tokens), "validation_tokens": len(valid_tokens),
                        "manifest": str(args.manifest),
                        "provenance": [
                            {k: r[k] for k in ("path", "source_name", "source_url", "retrieved_at", "license_id", "license_url", "attribution", "category", "reviewed_by", "sha256")}
                            for r in train_rows + valid_rows
                        ],
                        "live_model_modified": False,
                    },
                }, output)

    if not output.is_file():
        raise RuntimeError("no valid candidate checkpoint was saved")
    print(json.dumps({
        "status": "candidate_saved_not_promoted",
        "checkpoint": str(output),
        "steps_requested": args.steps,
        "steps_completed": history[-1]["step"] if history else 0,
        "best_validation_loss": best_val,
        "live_model_modified": False,
        "promotion": "manual review and independent evaluation required",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
