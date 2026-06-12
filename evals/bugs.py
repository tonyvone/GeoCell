"""Catalog of 50 seeded bugs across six categories.

Each bug is a *literal* corruption applied to a clean source file, plus
metadata. Bugs are organized into sibling GROUPS: members of a group
share the exact same defect transformation but live in different
functions with different identifiers (e.g. `per_page` vs `per_shard`).

The eval splits each group into TRAIN members (an LLM authors a verified,
generalized fix that GeoCell stores) and TEST members (near-repeats that
GeoCell must fix from memory, with no LLM call). A fix is generalized by
abstracting the buggy function's own identifiers, so a repair learned on
one sibling transfers to the others — and geometric recall decides which
learned repair matches a new symptom.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Bug:
    id: str
    category: str
    group: str
    file: str
    func: str            # the function the bug lives in (its identifiers get abstracted)
    broken: str          # literal substring to inject (must be unique in the file)
    clean: str           # the original substring it replaces
    symptom: str         # natural-language description; what GeoCell encodes
    test: str            # pytest node that must pass after a fix
    params: List[str] = field(default_factory=list)


def _b(id, category, group, file, func, broken, clean, symptom, test, params=()):
    return Bug(id, category, group, file, func, broken, clean, symptom, test, list(params))


BUGS: List[Bug] = [
    # ---- auth: inverted compare_digest guard -----------------------------
    _b("auth-cmp-1", "auth", "auth_inverted_compare", "service/auth.py", "verify_password",
       "    if hmac.compare_digest(stored, supplied_hash):\n        return False",
       "    if not hmac.compare_digest(stored, supplied_hash):\n        return False",
       "auth bypass: verify_password inverted hmac.compare_digest check rejects valid passwords",
       "tests/test_service.py::test_verify_password", ["stored", "supplied_hash"]),
    _b("auth-cmp-2", "auth", "auth_inverted_compare", "service/auth.py", "verify_api_key",
       "    if hmac.compare_digest(stored, supplied_key):\n        return False",
       "    if not hmac.compare_digest(stored, supplied_key):\n        return False",
       "auth bypass: verify_api_key inverted hmac.compare_digest check rejects valid api keys",
       "tests/test_service.py::test_verify_api_key", ["stored", "supplied_key"]),
    _b("auth-cmp-3", "auth", "auth_inverted_compare", "service/auth.py", "verify_webhook_sig",
       "    if hmac.compare_digest(expected_sig, supplied_sig):\n        return False",
       "    if not hmac.compare_digest(expected_sig, supplied_sig):\n        return False",
       "auth bypass: verify_webhook_sig inverted hmac.compare_digest check rejects valid signatures",
       "tests/test_service.py::test_verify_webhook_sig", ["expected_sig", "supplied_sig"]),

    # ---- auth: flipped expiry comparison ---------------------------------
    _b("auth-exp-1", "auth", "auth_flipped_expiry", "service/auth.py", "session_expired",
       "    if now < expires:", "    if now > expires:",
       "session_expired uses wrong comparison operator, treats fresh sessions as expired",
       "tests/test_service.py::test_session_expired", ["now", "expires"]),
    _b("auth-exp-2", "auth", "auth_flipped_expiry", "service/auth.py", "token_expired",
       "    if now < expires:", "    if now > expires:",
       "token_expired uses wrong comparison operator, treats fresh tokens as expired",
       "tests/test_service.py::test_token_expired", ["now", "expires"]),
    _b("auth-exp-3", "auth", "auth_flipped_expiry", "service/auth.py", "refresh_needed",
       "    if now < expires:", "    if now > expires:",
       "refresh_needed uses wrong comparison operator, triggers refresh too early",
       "tests/test_service.py::test_refresh_needed", ["now", "expires"]),

    # ---- auth: inverted role check ---------------------------------------
    _b("auth-role-1", "auth", "auth_inverted_role", "service/auth.py", "require_admin",
       '    if role == "admin":', '    if role != "admin":',
       "require_admin inverted role check raises for admins and allows non-admins",
       "tests/test_service.py::test_require_admin", ["role"]),
    _b("auth-role-2", "auth", "auth_inverted_role", "service/auth.py", "ensure_admin_for_delete",
       '    if role == "admin":', '    if role != "admin":',
       "ensure_admin_for_delete inverted role check lets non-admins delete",
       "tests/test_service.py::test_ensure_admin_for_delete", ["role"]),
    _b("auth-role-3", "auth", "auth_inverted_role", "service/auth.py", "ensure_admin_for_export",
       '    if role == "admin":', '    if role != "admin":',
       "ensure_admin_for_export inverted role check lets non-admins export",
       "tests/test_service.py::test_ensure_admin_for_export", ["role"]),

    # ---- auth: inverted presence check -----------------------------------
    _b("auth-tok-1", "auth", "auth_inverted_presence", "service/auth.py", "require_token",
       "    if token:\n        raise PermissionError", "    if not token:\n        raise PermissionError",
       "require_token inverted presence check rejects valid tokens and allows empty ones",
       "tests/test_service.py::test_require_token", ["token"]),
    _b("auth-tok-2", "auth", "auth_inverted_presence", "service/auth.py", "require_session",
       "    if token:\n        raise PermissionError", "    if not token:\n        raise PermissionError",
       "require_session inverted presence check rejects valid sessions and allows empty ones",
       "tests/test_service.py::test_require_session", ["token"]),
    _b("auth-tok-3", "auth", "auth_inverted_presence", "service/auth.py", "require_csrf",
       "    if token:\n        raise PermissionError", "    if not token:\n        raise PermissionError",
       "require_csrf inverted presence check rejects valid csrf tokens and allows empty ones",
       "tests/test_service.py::test_require_csrf", ["token"]),

    # ---- API routing: wrong strip direction ------------------------------
    _b("route-1", "api_routing", "route_strip", "service/routes.py", "normalize_user_path",
       'return path.lstrip("/")', 'return path.rstrip("/")',
       "normalize_user_path uses lstrip instead of rstrip, mangles route paths",
       "tests/test_service.py::test_normalize_user_path", ["path"]),
    _b("route-2", "api_routing", "route_strip", "service/routes.py", "normalize_order_path",
       'return path.lstrip("/")', 'return path.rstrip("/")',
       "normalize_order_path uses lstrip instead of rstrip, mangles order route paths",
       "tests/test_service.py::test_normalize_order_path", ["path"]),
    _b("route-3", "api_routing", "route_strip", "service/routes.py", "normalize_report_path",
       'return path.lstrip("/")', 'return path.rstrip("/")',
       "normalize_report_path uses lstrip instead of rstrip, mangles report route paths",
       "tests/test_service.py::test_normalize_report_path", ["path"]),

    # ---- API routing: missing-route handling -----------------------------
    _b("route-disp-1", "api_routing", "route_missing", "service/routes.py", "dispatch",
       "    handler = ROUTES.get((method, path))\n    if handler is None:\n        return None",
       '    handler = ROUTES.get((method, path))\n    if handler is None:\n        raise LookupError(f"no route {method} {path}")',
       "dispatch returns None instead of raising on unknown route, hiding 404s",
       "tests/test_service.py::test_dispatch_users", ["method", "path"]),

    # ---- type errors: wrong coercion -------------------------------------
    _b("type-parse-1", "type_errors", "type_int_coerce", "service/typesafe.py", "parse_age",
       "    return str(raw)", "    return int(raw)",
       "parse_age returns a string instead of int, downstream arithmetic breaks",
       "tests/test_service.py::test_parse_age", ["raw"]),
    _b("type-parse-2", "type_errors", "type_int_coerce", "service/typesafe.py", "parse_quantity",
       "    return str(raw)", "    return int(raw)",
       "parse_quantity returns a string instead of int, quantity math breaks",
       "tests/test_service.py::test_parse_quantity", ["raw"]),
    _b("type-parse-3", "type_errors", "type_int_coerce", "service/typesafe.py", "parse_limit",
       "    return str(raw)", "    return int(raw)",
       "parse_limit returns a string instead of int, pagination limit math breaks",
       "tests/test_service.py::test_parse_limit", ["raw"]),

    # ---- type errors: missing str() on concat ----------------------------
    _b("type-key-1", "type_errors", "type_str_concat", "service/typesafe.py", "cache_key_user",
       "    return prefix + ident", "    return prefix + str(ident)",
       "cache_key_user concatenates int ident without str(), raises TypeError",
       "tests/test_service.py::test_cache_key_user", ["prefix", "ident"]),
    _b("type-key-2", "type_errors", "type_str_concat", "service/typesafe.py", "cache_key_order",
       "    return prefix + ident", "    return prefix + str(ident)",
       "cache_key_order concatenates int ident without str(), raises TypeError",
       "tests/test_service.py::test_cache_key_order", ["prefix", "ident"]),
    _b("type-key-3", "type_errors", "type_str_concat", "service/typesafe.py", "cache_key_report",
       "    return prefix + ident", "    return prefix + str(ident)",
       "cache_key_report concatenates int ident without str(), raises TypeError",
       "tests/test_service.py::test_cache_key_report", ["prefix", "ident"]),

    # ---- config: unsafe dict access --------------------------------------
    _b("cfg-1", "config", "config_keyerror", "service/config.py", "get_timeout",
       '    return settings["timeout"]', '    return settings.get("timeout", 30)',
       "get_timeout indexes settings directly, KeyError when timeout missing",
       "tests/test_service.py::test_get_timeout", ["settings"]),
    _b("cfg-2", "config", "config_keyerror", "service/config.py", "get_retries",
       '    return settings["retries"]', '    return settings.get("retries", 3)',
       "get_retries indexes settings directly, KeyError when retries missing",
       "tests/test_service.py::test_get_retries", ["settings"]),
    _b("cfg-3", "config", "config_keyerror", "service/config.py", "get_pool_size",
       '    return settings["pool_size"]', '    return settings.get("pool_size", 5)',
       "get_pool_size indexes settings directly, KeyError when pool_size missing",
       "tests/test_service.py::test_get_pool_size", ["settings"]),
    _b("cfg-4", "config", "config_keyerror", "service/config.py", "get_debug",
       '    return settings["debug"]', '    return settings.get("debug", False)',
       "get_debug indexes settings directly, KeyError when debug missing",
       "tests/test_service.py::test_get_debug", ["settings"]),
    _b("cfg-5", "config", "config_keyerror", "service/config.py", "get_strict_mode",
       '    return settings["strict_mode"]', '    return settings.get("strict_mode", False)',
       "get_strict_mode indexes settings directly, KeyError when strict_mode missing",
       "tests/test_service.py::test_get_strict_mode", ["settings"]),
    _b("cfg-6", "config", "config_keyerror", "service/config.py", "get_telemetry",
       '    return settings["telemetry"]', '    return settings.get("telemetry", False)',
       "get_telemetry indexes settings directly, KeyError when telemetry missing",
       "tests/test_service.py::test_get_telemetry", ["settings"]),

    # ---- database: missing commit ----------------------------------------
    _b("db-commit-1", "database", "db_missing_commit", "service/db.py", "add_user",
       '        cur.execute("INSERT INTO users (name) VALUES (?)", params)\n',
       '        cur.execute("INSERT INTO users (name) VALUES (?)", params)\n        self.conn.commit()\n',
       "add_user never commits the INSERT, the row is lost",
       "tests/test_service.py::test_add_get_user", ["name"]),
    _b("db-commit-2", "database", "db_missing_commit", "service/db.py", "add_order",
       '        cur.execute("INSERT INTO orders (item) VALUES (?)", params)\n',
       '        cur.execute("INSERT INTO orders (item) VALUES (?)", params)\n        self.conn.commit()\n',
       "add_order never commits the INSERT, the row is lost",
       "tests/test_service.py::test_add_get_order", ["item"]),
    _b("db-commit-3", "database", "db_missing_commit", "service/db.py", "add_event",
       '        cur.execute("INSERT INTO events (kind) VALUES (?)", params)\n',
       '        cur.execute("INSERT INTO events (kind) VALUES (?)", params)\n        self.conn.commit()\n',
       "add_event never commits the INSERT, the row is lost",
       "tests/test_service.py::test_add_get_event", ["kind"]),

    # ---- database: parameter ignored in WHERE ----------------------------
    _b("db-where-1", "database", "db_ignored_param", "service/db.py", "get_user",
       '        cur.execute("SELECT id, name FROM users WHERE id = 0", params)',
       '        cur.execute("SELECT id, name FROM users WHERE id = ?", params)',
       "get_user hardcodes WHERE id = 0 and ignores the bound parameter",
       "tests/test_service.py::test_add_get_user", ["user_id"]),
    _b("db-where-2", "database", "db_ignored_param", "service/db.py", "get_order",
       '        cur.execute("SELECT id, item FROM orders WHERE id = 0", params)',
       '        cur.execute("SELECT id, item FROM orders WHERE id = ?", params)',
       "get_order hardcodes WHERE id = 0 and ignores the bound parameter",
       "tests/test_service.py::test_add_get_order", ["order_id"]),
    _b("db-where-3", "database", "db_ignored_param", "service/db.py", "get_event",
       '        cur.execute("SELECT id, kind FROM events WHERE id = 0", params)',
       '        cur.execute("SELECT id, kind FROM events WHERE id = ?", params)',
       "get_event hardcodes WHERE id = 0 and ignores the bound parameter",
       "tests/test_service.py::test_add_get_event", ["event_id"]),

    # ---- dependency misuse: floor div instead of ceil --------------------
    _b("dep-ceil-1", "dependency", "dep_floor_div", "service/deps.py", "page_count",
       "    return total // per_page", "    return math.ceil(total / per_page)",
       "page_count uses floor division instead of math.ceil, undercounts pages",
       "tests/test_service.py::test_page_count", ["total", "per_page"]),
    _b("dep-ceil-2", "dependency", "dep_floor_div", "service/deps.py", "shard_count",
       "    return total // per_shard", "    return math.ceil(total / per_shard)",
       "shard_count uses floor division instead of math.ceil, undercounts shards",
       "tests/test_service.py::test_shard_count", ["total", "per_shard"]),
    _b("dep-ceil-3", "dependency", "dep_floor_div", "service/deps.py", "batch_count",
       "    return total // per_batch", "    return math.ceil(total / per_batch)",
       "batch_count uses floor division instead of math.ceil, undercounts batches",
       "tests/test_service.py::test_batch_count", ["total", "per_batch"]),

    # ---- dependency misuse: json without sort_keys -----------------------
    _b("dep-json-1", "dependency", "dep_json_sort", "service/deps.py", "serialize_user",
       "    return json.dumps(record)", "    return json.dumps(record, sort_keys=True)",
       "serialize_user drops sort_keys, output key order is nondeterministic",
       "tests/test_service.py::test_serialize_user", ["record"]),
    _b("dep-json-2", "dependency", "dep_json_sort", "service/deps.py", "serialize_event",
       "    return json.dumps(record)", "    return json.dumps(record, sort_keys=True)",
       "serialize_event drops sort_keys, output key order is nondeterministic",
       "tests/test_service.py::test_serialize_event", ["record"]),
    _b("dep-json-3", "dependency", "dep_json_sort", "service/deps.py", "serialize_order",
       "    return json.dumps(record)", "    return json.dumps(record, sort_keys=True)",
       "serialize_order drops sort_keys, output key order is nondeterministic",
       "tests/test_service.py::test_serialize_order", ["record"]),

    # ---- dependency misuse: wrong sort direction -------------------------
    _b("dep-sort-1", "dependency", "dep_sort_dir", "service/deps.py", "newest_first",
       '    return sorted(items, key=lambda r: r["ts"])',
       '    return sorted(items, key=lambda r: r["ts"], reverse=True)',
       "newest_first forgets reverse=True, returns oldest-first order",
       "tests/test_service.py::test_newest_first", ["items"]),

    # ---- dependency misuse: weak hash ------------------------------------
    _b("dep-hash-1", "dependency", "dep_weak_hash", "service/deps.py", "checksum_payload",
       "    return hashlib.md5(data).hexdigest()", "    return hashlib.sha256(data).hexdigest()",
       "checksum_payload uses md5 instead of sha256, wrong digest and weak hash",
       "tests/test_service.py::test_checksum_payload", ["data"]),
    _b("dep-hash-2", "dependency", "dep_weak_hash", "service/deps.py", "checksum_file",
       "    return hashlib.md5(data).hexdigest()", "    return hashlib.sha256(data).hexdigest()",
       "checksum_file uses md5 instead of sha256, wrong digest and weak hash",
       "tests/test_service.py::test_checksum_file", ["data"]),
    _b("dep-hash-3", "dependency", "dep_weak_hash", "service/deps.py", "checksum_token",
       "    return hashlib.md5(data).hexdigest()", "    return hashlib.sha256(data).hexdigest()",
       "checksum_token uses md5 instead of sha256, wrong digest and weak hash",
       "tests/test_service.py::test_checksum_token", ["data"]),

    # ---- additional siblings (extend groups to 50 total) -----------------
    _b("type-parse-4", "type_errors", "type_int_coerce", "service/typesafe.py", "parse_offset",
       "    return str(raw)", "    return int(raw)",
       "parse_offset returns a string instead of int, offset arithmetic breaks",
       "tests/test_service.py::test_parse_offset", ["raw"]),
    _b("type-parse-5", "type_errors", "type_int_coerce", "service/typesafe.py", "parse_page",
       "    return str(raw)", "    return int(raw)",
       "parse_page returns a string instead of int, page arithmetic breaks",
       "tests/test_service.py::test_parse_page", ["raw"]),
    _b("type-key-4", "type_errors", "type_str_concat", "service/typesafe.py", "cache_key_session",
       "    return prefix + ident", "    return prefix + str(ident)",
       "cache_key_session concatenates int ident without str(), raises TypeError",
       "tests/test_service.py::test_cache_key_session", ["prefix", "ident"]),
    _b("type-key-5", "type_errors", "type_str_concat", "service/typesafe.py", "cache_key_token",
       "    return prefix + ident", "    return prefix + str(ident)",
       "cache_key_token concatenates int ident without str(), raises TypeError",
       "tests/test_service.py::test_cache_key_token", ["prefix", "ident"]),
    _b("cfg-7", "config", "config_keyerror", "service/config.py", "get_max_conn",
       '    return settings["max_conn"]', '    return settings.get("max_conn", 10)',
       "get_max_conn indexes settings directly, KeyError when max_conn missing",
       "tests/test_service.py::test_get_max_conn", ["settings"]),
    _b("cfg-8", "config", "config_keyerror", "service/config.py", "get_ttl",
       '    return settings["ttl"]', '    return settings.get("ttl", 60)',
       "get_ttl indexes settings directly, KeyError when ttl missing",
       "tests/test_service.py::test_get_ttl", ["settings"]),
    _b("route-4", "api_routing", "route_strip", "service/routes.py", "normalize_event_path",
       'return path.lstrip("/")', 'return path.rstrip("/")',
       "normalize_event_path uses lstrip instead of rstrip, mangles event route paths",
       "tests/test_service.py::test_normalize_event_path", ["path"]),
    _b("route-5", "api_routing", "route_strip", "service/routes.py", "normalize_invoice_path",
       'return path.lstrip("/")', 'return path.rstrip("/")',
       "normalize_invoice_path uses lstrip instead of rstrip, mangles invoice route paths",
       "tests/test_service.py::test_normalize_invoice_path", ["path"]),
    _b("dep-ceil-4", "dependency", "dep_floor_div", "service/deps.py", "chunk_count",
       "    return total // per_chunk", "    return math.ceil(total / per_chunk)",
       "chunk_count uses floor division instead of math.ceil, undercounts chunks",
       "tests/test_service.py::test_chunk_count", ["total", "per_chunk"]),
    _b("dep-json-4", "dependency", "dep_json_sort", "service/deps.py", "serialize_session",
       "    return json.dumps(record)", "    return json.dumps(record, sort_keys=True)",
       "serialize_session drops sort_keys, output key order is nondeterministic",
       "tests/test_service.py::test_serialize_session", ["record"]),
    _b("dep-sort-2", "dependency", "dep_extra_reverse", "service/deps.py", "oldest_first",
       '    return sorted(items, key=lambda r: r["ts"], reverse=True)',
       '    return sorted(items, key=lambda r: r["ts"])',
       "oldest_first wrongly sorts reverse=True, returns newest-first order",
       "tests/test_service.py::test_oldest_first", ["items"]),
    _b("dep-sort-3", "dependency", "dep_extra_reverse", "service/deps.py", "by_timestamp",
       '    return sorted(items, key=lambda r: r["ts"], reverse=True)',
       '    return sorted(items, key=lambda r: r["ts"])',
       "by_timestamp wrongly sorts reverse=True, returns newest-first order",
       "tests/test_service.py::test_by_timestamp", ["items"]),
]


def by_group():
    groups = {}
    for b in BUGS:
        groups.setdefault(b.group, []).append(b)
    return groups


assert len({b.id for b in BUGS}) == len(BUGS), "duplicate bug ids"
