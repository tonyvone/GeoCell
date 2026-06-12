"""Behavioral test suite for the sample service.

This is the ground-truth pass/fail oracle for the coding eval: every
seeded bug breaks at least one assertion here, and a fix is only counted
if the relevant test goes green again. No mocks of correctness.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from service import auth, config, db, deps, routes, typesafe  # noqa: E402


# ---- auth ----------------------------------------------------------------

def test_verify_password():
    assert auth.verify_password("alice", "h-2b8c") is True
    assert auth.verify_password("alice", "wrong") is False

def test_verify_api_key():
    assert auth.verify_api_key("svc-reporting", "k-9f2a") is True
    assert auth.verify_api_key("svc-reporting", "nope") is False

def test_verify_webhook_sig():
    assert auth.verify_webhook_sig("sig-abc", "sig-abc") is True
    assert auth.verify_webhook_sig("sig-abc", "sig-xyz") is False

def test_session_expired():
    assert auth.session_expired(100.0, 50.0) is True
    assert auth.session_expired(40.0, 50.0) is False

def test_token_expired():
    assert auth.token_expired(100.0, 50.0) is True
    assert auth.token_expired(40.0, 50.0) is False

def test_refresh_needed():
    assert auth.refresh_needed(100.0, 50.0) is True
    assert auth.refresh_needed(40.0, 50.0) is False

def test_require_admin():
    auth.require_admin("admin")
    try:
        auth.require_admin("user"); assert False
    except PermissionError:
        pass

def test_ensure_admin_for_delete():
    auth.ensure_admin_for_delete("admin")
    try:
        auth.ensure_admin_for_delete("user"); assert False
    except PermissionError:
        pass

def test_ensure_admin_for_export():
    auth.ensure_admin_for_export("admin")
    try:
        auth.ensure_admin_for_export("user"); assert False
    except PermissionError:
        pass

def test_require_token():
    auth.require_token("t")
    try:
        auth.require_token(""); assert False
    except PermissionError:
        pass

def test_require_session():
    auth.require_session("t")
    try:
        auth.require_session(""); assert False
    except PermissionError:
        pass

def test_require_csrf():
    auth.require_csrf("t")
    try:
        auth.require_csrf(""); assert False
    except PermissionError:
        pass


# ---- routing -------------------------------------------------------------

def test_normalize_user_path():
    assert routes.normalize_user_path("/users/") == "/users"

def test_normalize_order_path():
    assert routes.normalize_order_path("/orders/") == "/orders"

def test_normalize_report_path():
    assert routes.normalize_report_path("/reports/") == "/reports"

def test_normalize_event_path():
    assert routes.normalize_event_path("/events/") == "/events"

def test_normalize_invoice_path():
    assert routes.normalize_invoice_path("/invoices/") == "/invoices"

def test_dispatch_users():
    assert routes.dispatch("GET", "/users") == ["alice", "bob"]

def test_dispatch_orders():
    assert routes.dispatch("GET", "/orders") == ["o-1", "o-2"]


# ---- type safety ---------------------------------------------------------

def test_parse_age():
    v = typesafe.parse_age("42"); assert isinstance(v, int) and v == 42

def test_parse_quantity():
    v = typesafe.parse_quantity("7"); assert isinstance(v, int) and v == 7

def test_parse_limit():
    v = typesafe.parse_limit("10"); assert isinstance(v, int) and v == 10

def test_parse_offset():
    v = typesafe.parse_offset("8"); assert isinstance(v, int) and v == 8

def test_parse_page():
    v = typesafe.parse_page("2"); assert isinstance(v, int) and v == 2

def test_cache_key_user():
    assert typesafe.cache_key_user("u:", 5) == "u:5"

def test_cache_key_order():
    assert typesafe.cache_key_order("o:", 9) == "o:9"

def test_cache_key_report():
    assert typesafe.cache_key_report("r:", 3) == "r:3"

def test_cache_key_session():
    assert typesafe.cache_key_session("s:", 4) == "s:4"

def test_cache_key_token():
    assert typesafe.cache_key_token("t:", 6) == "t:6"


# ---- config --------------------------------------------------------------

def test_get_timeout():
    assert config.get_timeout({}) == 30
    assert config.get_timeout({"timeout": 5}) == 5

def test_get_retries():
    assert config.get_retries({}) == 3

def test_get_pool_size():
    assert config.get_pool_size({}) == 5

def test_get_debug():
    assert config.get_debug({}) is False

def test_get_strict_mode():
    assert config.get_strict_mode({}) is False

def test_get_telemetry():
    assert config.get_telemetry({}) is False

def test_get_max_conn():
    assert config.get_max_conn({}) == 10

def test_get_ttl():
    assert config.get_ttl({}) == 60


# ---- database ------------------------------------------------------------

def test_add_get_user():
    s = db.Store(); s.add_user("zoe")
    assert s.get_user(1) == (1, "zoe")

def test_add_get_order():
    s = db.Store(); s.add_order("widget")
    assert s.get_order(1) == (1, "widget")

def test_add_get_event():
    s = db.Store(); s.add_event("login")
    assert s.get_event(1) == (1, "login")


# ---- dependency usage ----------------------------------------------------

def test_page_count():
    assert deps.page_count(10, 3) == 4

def test_shard_count():
    assert deps.shard_count(10, 4) == 3

def test_batch_count():
    assert deps.batch_count(7, 2) == 4

def test_chunk_count():
    assert deps.chunk_count(9, 4) == 3

def test_serialize_user():
    assert deps.serialize_user({"b": 1, "a": 2}) == '{"a": 2, "b": 1}'

def test_serialize_event():
    assert deps.serialize_event({"b": 1, "a": 2}) == '{"a": 2, "b": 1}'

def test_serialize_order():
    assert deps.serialize_order({"b": 1, "a": 2}) == '{"a": 2, "b": 1}'

def test_serialize_session():
    assert deps.serialize_session({"b": 1, "a": 2}) == '{"a": 2, "b": 1}'

def test_newest_first():
    out = deps.newest_first([{"ts": 1}, {"ts": 3}, {"ts": 2}])
    assert [r["ts"] for r in out] == [3, 2, 1]

def test_oldest_first():
    out = deps.oldest_first([{"ts": 3}, {"ts": 1}, {"ts": 2}])
    assert [r["ts"] for r in out] == [1, 2, 3]

def test_by_timestamp():
    out = deps.by_timestamp([{"ts": 3}, {"ts": 1}])
    assert [r["ts"] for r in out] == [1, 3]

def test_checksum_payload():
    import hashlib
    assert deps.checksum_payload(b"x") == hashlib.sha256(b"x").hexdigest()

def test_checksum_file():
    import hashlib
    assert deps.checksum_file(b"x") == hashlib.sha256(b"x").hexdigest()

def test_checksum_token():
    import hashlib
    assert deps.checksum_token(b"x") == hashlib.sha256(b"x").hexdigest()
