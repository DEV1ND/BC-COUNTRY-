"""Supervised fine-tuning (SFT) for the isolated Quantum candidate.

Input: reviewed JSONL records with {"prompt": "...", "response": "...",
"approved": true, "source": "...", "license": "..."}.
Only assistant response tokens contribute to loss. Outputs a candidate file;
never overwrites the serving checkpoint.
"""
import argparse, hashlib, json, math, os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from bpe_tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Candidate(nn.Module):
    def __init__(self, vocab, width, heads, layers, block):
        super().__init__()
        self.token = nn.Embedding(vocab, width)
        self.position = nn.Embedding(block, width)
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=heads,
            dim_feedforward=width*4, dropout=0.1, batch_first=True,
            norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocab)
        self.register_buffer("causal", torch.triu(torch.ones(block, block, dtype=torch.bool), diagonal=1))
    def forward(self, ids):
        n = ids.shape[1]
        x = self.token(ids) + self.position(torch.arange(n, device=ids.device))
        return self.output(self.norm(self.encoder(x, mask=self.causal[:n,:n])))

def read_records(path):
    rows, seen = [], set()
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip(): continue
        obj = json.loads(line)
        if obj.get("approved") is not True:
            continue
        if not all(isinstance(obj.get(k), str) and obj[k].strip() for k in ("prompt","response","source","license")):
            raise ValueError(f"JSONL line {line_no}: prompt/response/source/license required")
        key = hashlib.sha256((obj["prompt"].strip()+"\n"+obj["response"].strip()).encode()).hexdigest()
        if key in seen: continue
        seen.add(key)
        rows.append(obj)
    if len(rows) < 20:
        raise ValueError(f"At least 20 approved dialogue examples required; got {len(rows)}")
    return rows

def encode_rows(rows, tok, block):
    out=[]
    for row in rows:
        prefix = "사용자: " + row["prompt"].strip() + "\nQuantum: "
        full = prefix + row["response"].strip() + "\n"
        ids = tok.encode(full)
        prefix_n = min(len(tok.encode(prefix)), len(ids))
        if len(ids) > block:
            ids = ids[-block:]
            prefix_n = max(0, prefix_n - (len(tok.encode(full))-block))
        if len(ids) >= 8 and prefix_n < len(ids):
            labels = [-100]*prefix_n + ids[prefix_n:]
            out.append((ids, labels))
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data", default="quantum_sft_train.jsonl")
    p.add_argument("--checkpoint", default=os.getenv("QUANTUM_CHECKPOINT_PATH","bpe_candidate_checkpoint.pt"))
    p.add_argument("--output", default="quantum_sft_candidate.pt")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=2)
    args=p.parse_args()
    base=Path(args.checkpoint)
    if not base.exists(): raise FileNotFoundError("Base pretraining checkpoint not found.")
    rows=read_records(args.data)
    payload=torch.load(base,map_location="cpu",weights_only=False)
    arch=payload["architecture"]; block=int(arch.get("block",256))
    tok=BPETokenizer.from_dict(payload["tokenizer"])
    encoded=encode_rows(rows,tok,block)
    if len(encoded)<16: raise ValueError("Not enough usable SFT sequences.")
    # Deterministic prompt-level holdout; no near-duplicate train/validation overlap.
    ordered=sorted(encoded,key=lambda x: hashlib.sha256(str(x[0][:24]).encode()).hexdigest())
    split=max(1,min(len(ordered)-1,int(len(ordered)*0.9)))
    train,valid=ordered[:split],ordered[split:]
    model=Candidate(len(tok.vocab),int(arch["width"]),int(arch["heads"]),int(arch["layers"]),block).to(DEVICE)
    model.load_state_dict(payload["state_dict"],strict=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=0.01)
    def evaluate():
        model.eval(); total=0.; count=0
        with torch.no_grad():
            for ids,labels in valid:
                x=torch.tensor(ids[:-1],device=DEVICE).unsqueeze(0)
                y=torch.tensor(labels[1:],device=DEVICE)
                logits=model(x)[0]
                mask=y.ne(-100)
                if mask.any():
                    total += F.cross_entropy(logits[mask],y[mask],reduction="sum").item()
                    count += int(mask.sum())
        return total/max(1,count)
    baseline=evaluate(); best=baseline; best_state=None
    model.train()
    steps=max(20,min(2000,args.steps))
    for step in range(steps):
        batch=[train[torch.randint(len(train),(1,)).item()] for _ in range(max(1,min(8,args.batch_size)))]
        width=max(len(x[0]) for x in batch)-1
        xs=torch.zeros((len(batch),width),dtype=torch.long)
        ys=torch.full((len(batch),width),-100,dtype=torch.long)
        for i,(ids,labels) in enumerate(batch):
            n=len(ids)-1; xs[i,:n]=torch.tensor(ids[:-1]); ys[i,:n]=torch.tensor(labels[1:])
        xs=xs.to(DEVICE); ys=ys.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits=model(xs)
        loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),ys.reshape(-1),ignore_index=-100)
        if not torch.isfinite(loss): raise RuntimeError("Non-finite SFT loss")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
        if (step+1)%20==0 or step+1==steps:
            score=evaluate()
            if math.isfinite(score) and score<best:
                best=score
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            model.train()
    if best_state is None or best >= baseline*0.999:
        raise RuntimeError("SFT candidate rejected: validation loss did not improve.")
    model.load_state_dict(best_state)
    report={"stage":"supervised_finetuning","status":"candidate_ready_for_safety_eval",
        "approved_examples":len(rows),"train_examples":len(train),"validation_examples":len(valid),
        "steps":steps,"baseline_validation_loss":baseline,"candidate_validation_loss":best,
        "candidate":args.output,"serving_checkpoint_changed":False,
        "source_policy":"Only approved records with source and license metadata."}
    torch.save({"state_dict":model.state_dict(),"tokenizer":tok.to_dict(),"architecture":arch,
        "steps":steps,"training_metadata":report},args.output)
    Path("quantum_sft_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
