"""Input normalization and deterministic top-k filtering for Quantum inference."""
import torch


def normalize_prompt(value):
    """Trim user input and reject empty or unsafe control-character prompts."""
    if not isinstance(value, str):
        raise ValueError("prompt는 문자열이어야 합니다.")
    prompt = value.strip()
    if not prompt:
        raise ValueError("prompt는 공백만 입력할 수 없습니다.")
    if any(
        ord(char) == 0
        or (ord(char) < 32 and char not in "\t\n\r")
        or 127 <= ord(char) <= 159
        for char in prompt
    ):
        raise ValueError("prompt에 허용되지 않는 제어 문자가 포함되어 있습니다.")
    return prompt


def apply_top_k_logits(logits, top_k):
    """Keep only the highest-k logits; zero or oversized k leaves logits unchanged."""
    if not torch.is_tensor(logits) or logits.ndim != 1 or logits.numel() == 0:
        raise ValueError("logits는 비어 있지 않은 1차원 Tensor여야 합니다.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k는 0 이상의 정수여야 합니다.")
    if torch.isnan(logits).any() or torch.isposinf(logits).any():
        raise ValueError("logits에 NaN 또는 양의 무한대가 포함되어 있습니다.")
    if not torch.isfinite(logits).any():
        raise ValueError("logits에 유효한 후보가 없습니다.")
    if top_k == 0 or top_k >= logits.numel():
        return logits.clone()
    values, indices = torch.topk(logits, top_k)
    filtered = torch.full_like(logits, -float("inf"))
    filtered.scatter_(0, indices, values)
    return filtered
