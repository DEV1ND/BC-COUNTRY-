"""Deterministic BPE tokenizer candidate. Does not replace the live model tokenizer."""
import json
from collections import Counter
from pathlib import Path

class BPETokenizer:
    def __init__(self, merges=None, initial_symbols=None):
        raw_symbols = list(initial_symbols or [])
        if any(not isinstance(symbol, str) or len(symbol) != 1 for symbol in raw_symbols):
            raise ValueError("초기 토큰은 각각 정확히 한 글자여야 합니다.")
        if len(raw_symbols) != len(set(raw_symbols)):
            raise ValueError("초기 토큰에 중복 문자가 있습니다.")
        self.initial_symbols = raw_symbols
        self.merges = []
        available = {"<unk>", *raw_symbols}
        for index, pair in enumerate(merges or []):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"BPE 병합 {index}는 문자열 2개로 구성되어야 합니다.")
            left, right = pair
            if not isinstance(left, str) or not isinstance(right, str) or not left or not right:
                raise ValueError(f"BPE 병합 {index}에 비어 있거나 잘못된 토큰이 있습니다.")
            if left not in available or right not in available:
                raise ValueError(f"BPE 병합 {index}가 이전에 정의되지 않은 토큰을 참조합니다.")
            merged = left + right
            if merged in available:
                raise ValueError(f"BPE 병합 {index}가 이미 존재하는 토큰을 중복 생성합니다.")
            self.merges.append((left, right))
            available.add(merged)
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        symbols = {"<unk>"} | set(self.initial_symbols)
        for left, right in self.merges:
            symbols.add(left + right)
        self.vocab = ["<unk>"] + sorted(symbols - {"<unk>"})
        self.stoi = {token: i for i, token in enumerate(self.vocab)}
        self.itos = {i: token for token, i in self.stoi.items()}

    @classmethod
    def train(cls, texts, vocab_size=1024, min_pair_frequency=2, extra_alphabet_texts=None):
        words = []
        alphabet = set()
        for text in texts:
            chars = list(str(text))
            alphabet.update(chars)
            if chars:
                words.append(chars)
        # Validation text contributes only single-character coverage, never merge counts.
        # This prevents unseen Korean syllables from breaking tokenizer round-trip checks.
        for text in (extra_alphabet_texts or ()):
            alphabet.update(list(str(text)))
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
        output = []
        for index in ids:
            try:
                token_id = int(index)
            except (TypeError, ValueError, OverflowError):
                output.append("�")
                continue
            output.append(self.itos.get(token_id, "�"))
        return "".join(output)

    def to_dict(self):
        return {"format": "juckex-bpe-v1", "initial_symbols": self.initial_symbols,
                "merges": [list(pair) for pair in self.merges], "vocab_size": len(self.vocab)}

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict):
            raise ValueError("BPE 토크나이저 데이터는 객체여야 합니다.")
        if payload.get("format") != "juckex-bpe-v1":
            raise ValueError("지원하지 않는 BPE 포맷입니다.")
        symbols = payload.get("initial_symbols")
        merges = payload.get("merges")
        if not isinstance(symbols, list) or not isinstance(merges, list):
            raise ValueError("BPE 초기 토큰과 병합 목록은 배열이어야 합니다.")
        tokenizer = cls(merges=merges, initial_symbols=symbols)
        declared_size = payload.get("vocab_size")
        if declared_size is not None:
            if isinstance(declared_size, bool) or not isinstance(declared_size, int):
                raise ValueError("BPE vocab_size는 정수여야 합니다.")
            if declared_size != len(tokenizer.vocab):
                raise ValueError(
                    f"BPE vocab_size 불일치: 선언={declared_size}, 실제={len(tokenizer.vocab)}"
                )
        return tokenizer

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


def encode_prompt_answer(prompt, answer, tokenizer, suffix="\n\n"):
    """Encode prompt and target separately so BPE cannot merge across their boundary."""
    prompt_ids = tokenizer.encode(str(prompt))
    answer_ids = tokenizer.encode(str(answer) + suffix)
    return prompt_ids + answer_ids, len(prompt_ids)
