"""Fail-closed authentication policy for the isolated candidate inference API."""
import hmac


def generation_auth_error(api_token, authorization):
    """Return (HTTP status, safe message) when generation must be rejected."""
    if not isinstance(api_token, str) or not api_token:
        return 503, "후보 추론 서버 인증 설정이 완료되지 않았습니다."
    expected = "Bearer " + api_token
    if not isinstance(authorization, str) or not hmac.compare_digest(authorization.encode('utf-8'), expected.encode('utf-8')):
        return 401, "인증이 필요합니다."
    return None
