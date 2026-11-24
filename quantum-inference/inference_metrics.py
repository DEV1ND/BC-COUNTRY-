"""Small, dependency-free timing helpers for Quantum inference diagnostics."""
import math


def elapsed_ms(started_at, finished_at):
    """Return a monotonic elapsed duration in milliseconds."""
    values = (started_at, finished_at)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(float(value)) for value in values):
        raise ValueError("시간 값은 유한한 숫자여야 합니다.")
    if finished_at < started_at:
        raise ValueError("종료 시간은 시작 시간보다 빠를 수 없습니다.")
    return round((finished_at - started_at) * 1000.0, 3)


def tokens_per_second(token_count, duration_ms):
    """Return generated-token throughput; zero duration is reported as zero."""
    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
        raise ValueError("token_count는 0 이상의 정수여야 합니다.")
    if isinstance(duration_ms, bool) or not isinstance(duration_ms, (int, float)):
        raise ValueError("duration_ms는 0 이상의 유한한 숫자여야 합니다.")
    if not math.isfinite(float(duration_ms)) or duration_ms < 0:
        raise ValueError("duration_ms는 0 이상의 유한한 숫자여야 합니다.")
    if duration_ms == 0 or token_count == 0:
        return 0.0
    return round(token_count * 1000.0 / duration_ms, 3)
