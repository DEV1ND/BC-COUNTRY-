"""Input normalization and deterministic top-k selection for Quantum inference."""
import math


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


def select_top_k_indices(logit_values, top_k):
    """Return indices of the highest-k scores without depending on PyTorch."""
    if not isinstance(logit_values, (list, tuple)) or not logit_values:
        raise ValueError("logit_values는 비어 있지 않은 목록이어야 합니다.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k는 0 이상의 정수여야 합니다.")
    normalized = []
    has_finite = False
    for value in logit_values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("logit 값은 숫자여야 합니다.")
        if value != value or value == float("inf"):
            raise ValueError("logit에 NaN 또는 양의 무한대가 포함되어 있습니다.")
        if value != -float("inf"):
            has_finite = True
        normalized.append(float(value))
    if not has_finite:
        raise ValueError("logit에 유효한 후보가 없습니다.")
    if top_k == 0 or top_k >= len(normalized):
        return list(range(len(normalized)))
    return sorted(range(len(normalized)), key=lambda index: (-normalized[index], index))[:top_k]


def validate_sampling_probabilities(probability_values):
    """Reject invalid distributions instead of silently switching to greedy decoding."""
    if not isinstance(probability_values, (list, tuple)) or not probability_values:
        raise ValueError("확률 분포는 비어 있지 않은 목록이어야 합니다.")
    total = 0.0
    for value in probability_values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("확률 값은 숫자여야 합니다.")
        probability = float(value)
        if not math.isfinite(probability) or probability < 0:
            raise ValueError("확률 값은 유한한 0 이상의 숫자여야 합니다.")
        total += probability
    if not math.isfinite(total) or total <= 0:
        raise ValueError("확률 분포의 합은 0보다 커야 합니다.")
    return total
