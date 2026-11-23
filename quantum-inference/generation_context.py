"""Budget retrieved context without silently truncating the user's question."""


def _validate_token_ids(token_ids, label, tokenizer):
    if not isinstance(token_ids, (list, tuple)):
        raise ValueError(f"{label} 토큰 형식이 올바르지 않습니다.")
    vocab = getattr(tokenizer, "vocab", None)
    vocab_size = len(vocab) if isinstance(vocab, (list, tuple)) else None
    for token in token_ids:
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise ValueError(f"{label} 토큰 형식이 올바르지 않습니다.")
        if vocab_size is not None and token >= vocab_size:
            raise ValueError(f"{label} 토큰 ID가 어휘 범위를 벗어났습니다.")
    return list(token_ids)


def build_generation_context(tokenizer, knowledge_context, question, block_size=256, return_diagnostics=False):
    if not isinstance(return_diagnostics, bool):
        raise ValueError("return_diagnostics는 불리언이어야 합니다.")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 2:
        raise ValueError("block_size는 2 이상의 정수여야 합니다.")
    if not isinstance(knowledge_context, str) or not isinstance(question, str):
        raise ValueError("검색 문맥과 질문은 문자열이어야 합니다.")
    question_ids = _validate_token_ids(
        tokenizer.encode("질문:" + question.strip() + "\n답변:"),
        "질문", tokenizer,
    )
    if not question_ids:
        raise ValueError("질문 토큰을 생성하지 못했습니다.")
    if len(question_ids) > block_size:
        raise ValueError(
            f"질문이 Quantum 문맥 한도({block_size} 토큰)를 초과했습니다. 질문을 짧게 나눠 입력하세요."
        )

    context_ids = _validate_token_ids(
        tokenizer.encode(knowledge_context) if knowledge_context else [],
        "검색 문맥", tokenizer,
    )
    budget = block_size - len(question_ids)
    retrieved_context_tokens = len(context_ids)
    truncated = retrieved_context_tokens > budget
    if budget:
        context_ids = context_ids[-budget:]
    else:
        context_ids = []
    prompt_ids = context_ids + question_ids
    if not return_diagnostics:
        return prompt_ids, truncated
    diagnostics = {
        "block_size": block_size,
        "question_tokens": len(question_ids),
        "retrieved_context_tokens": retrieved_context_tokens,
        "included_context_tokens": len(context_ids),
        "dropped_context_tokens": retrieved_context_tokens - len(context_ids),
        "prompt_tokens": len(prompt_ids),
        "remaining_generation_capacity": max(0, block_size - len(prompt_ids)),
        "context_truncated": truncated,
        "context_policy": "keep_tail",
    }
    return prompt_ids, truncated, diagnostics
