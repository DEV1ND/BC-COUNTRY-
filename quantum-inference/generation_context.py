"""Budget retrieved context without silently truncating the user's question."""


def build_generation_context(tokenizer, knowledge_context, question, block_size=256):
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 2:
        raise ValueError("block_size는 2 이상의 정수여야 합니다.")
    if not isinstance(knowledge_context, str) or not isinstance(question, str):
        raise ValueError("검색 문맥과 질문은 문자열이어야 합니다.")
    question_ids = tokenizer.encode("질문:" + question.strip() + "\n답변:")
    if not isinstance(question_ids, (list, tuple)) or not question_ids:
        raise ValueError("질문 토큰을 생성하지 못했습니다.")
    if any(isinstance(token, bool) or not isinstance(token, int) for token in question_ids):
        raise ValueError("질문 토큰 형식이 올바르지 않습니다.")
    if len(question_ids) > block_size:
        raise ValueError(
            f"질문이 Quantum 문맥 한도({block_size} 토큰)를 초과했습니다. 질문을 짧게 나눠 입력하세요."
        )

    context_ids = tokenizer.encode(knowledge_context) if knowledge_context else []
    if not isinstance(context_ids, (list, tuple)):
        raise ValueError("검색 문맥 토큰 형식이 올바르지 않습니다.")
    if any(isinstance(token, bool) or not isinstance(token, int) for token in context_ids):
        raise ValueError("검색 문맥 토큰 형식이 올바르지 않습니다.")

    budget = block_size - len(question_ids)
    truncated = len(context_ids) > budget
    if budget:
        context_ids = list(context_ids[-budget:])
    else:
        context_ids = []
    return context_ids + list(question_ids), truncated
