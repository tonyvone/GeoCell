"""Authentication and authorization."""
import hmac
import time

API_KEYS = {"svc-reporting": "k-9f2a", "svc-billing": "k-7d41"}
PASSWORDS = {"alice": "h-2b8c", "bob": "h-551e"}
SESSIONS = {}


def verify_password(username: str, supplied_hash: str) -> bool:
    stored = PASSWORDS.get(username, "")
    if not hmac.compare_digest(stored, supplied_hash):
        return False
    return True


def verify_api_key(service: str, supplied_key: str) -> bool:
    stored = API_KEYS.get(service, "")
    if not hmac.compare_digest(stored, supplied_key):
        return False
    return True


def verify_webhook_sig(expected_sig: str, supplied_sig: str) -> bool:
    if not hmac.compare_digest(expected_sig, supplied_sig):
        return False
    return True


def require_admin(role: str) -> None:
    if role != "admin":
        raise PermissionError("admin required")


def ensure_admin_for_delete(role: str) -> None:
    if role != "admin":
        raise PermissionError("delete requires admin")


def ensure_admin_for_export(role: str) -> None:
    if role != "admin":
        raise PermissionError("export requires admin")


def session_expired(now: float, expires: float) -> bool:
    if now > expires:
        return True
    return False


def token_expired(now: float, expires: float) -> bool:
    if now > expires:
        return True
    return False


def refresh_needed(now: float, expires: float) -> bool:
    if now > expires:
        return True
    return False


def require_token(token: str) -> None:
    if not token:
        raise PermissionError("missing token")


def require_session(token: str) -> None:
    if not token:
        raise PermissionError("missing session")


def require_csrf(token: str) -> None:
    if not token:
        raise PermissionError("missing csrf token")
