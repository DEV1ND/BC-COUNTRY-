"""Fail-closed authentication policy for the isolated candidate inference API."""


def generation_auth_error(api_token, authorization):
    """Return (HTTP status, safe message) when generation must be rejected."""
    if not api_token:
        return 503, "후보 추론 서버 인증 설정이 완료되지 않았습니다."
    if authorization != "Bearer " + api_token:
        return 401, "인증이 필요합니다."
    return None
