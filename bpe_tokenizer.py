"""Deterministic BPE tokenizer candidate. Does not replace the live model tokenizer."""
import json
from collections import Counter
from pathlib import Path

class BPETokenizer:
    def __init__(self, merges=None, initial_symbols=None):
        self.initial_symbols = list(initial_symbols or [])
        self.merges = [tuple(x) for x in (merges or [])]
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        symbols = {"<unk>"} | set(self.initial_symbols)
        for left, right in self.merges:
            symbols.add(left + right)
        self.vocab = ["<unk>"] + sorted(symbols - {"<unk>"})
        self.stoi = {token: i for i, token in enumerate(self.vocab)}
        self.itos = {i: token for token, i in self.stoi.items()}

    @classmethod
    def train(cls, texts, vocab_size=1024, min_pair_frequency=2):
        words = []
        alphabet = set()
        for text in texts:
            chars = list(str(text))
            alphabet.update(chars)
            if chars:
                words.append(chars)
        merges = []
        target = max(len(alphabet) + 1, int(vocab_size))
        while len(alphabet) + len(merges) + 1 < target:
            counts = Counter()
            for word in words:
                counts.update(zip(word, word[1:]))
            if not counts:
                break
            pair, frequency = min(counts.items(), key=lambda item: (-item[1], item[0]))
            if frequency < min_pair_frequency:
                break
            merges.append(pair)
            merged = pair[0] + pair[1]
            updated = []
            for word in words:
                out, i = [], 0
                while i < len(word):
                    if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
                        out.append(merged)
                        i += 2
                    else:
                        out.append(word[i])
                        i += 1
                updated.append(out)
            words = updated
        return cls(merges=merges, initial_symbols=sorted(alphabet))

    def encode(self, text):
        pieces = list(str(text))
        while len(pieces) > 1:
            candidates = [(self.ranks[(pieces[i], pieces[i + 1])], i)
                          for i in range(len(pieces) - 1)
                          if (pieces[i], pieces[i + 1]) in self.ranks]
            if not candidates:
                break
            _, index = min(candidates)
            pieces[index:index + 2] = [pieces[index] + pieces[index + 1]]
        return [self.stoi.get(piece, 0) for piece in pieces]

    def decode(self, ids):
        return "".join(self.itos.get(int(index), "") if int(index) else "�" for index in ids)

    def to_dict(self):
        return {"format": "juckex-bpe-v1", "initial_symbols": self.initial_symbols,
                "merges": [list(pair) for pair in self.merges], "vocab_size": len(self.vocab)}

    @classmethod
    def from_dict(cls, payload):
        if payload.get("format") != "juckex-bpe-v1":
            raise ValueError("지원하지 않는 BPE 포맷입니다.")
        return cls(payload.get("merges", []), payload.get("initial_symbols", []))

def load_approved_texts(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("검수 데이터는 배열이어야 합니다.")
    texts = []
    for row in rows:
        if not isinstance(row, dict) or row.get("approved") is not True:
            continue
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if len(question) >= 2 and len(answer) >= 8:
            texts.extend([question, answer])
    if not texts:
        raise ValueError("사용 가능한 승인 데이터가 없습니다.")
    return texts
