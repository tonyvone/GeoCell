"""Type coercion at API boundaries."""


def parse_age(raw) -> int:
    return int(raw)


def parse_quantity(raw) -> int:
    return int(raw)


def parse_limit(raw) -> int:
    return int(raw)


def parse_offset(raw) -> int:
    return int(raw)


def parse_page(raw) -> int:
    return int(raw)


def cache_key_user(prefix: str, ident) -> str:
    return prefix + str(ident)


def cache_key_order(prefix: str, ident) -> str:
    return prefix + str(ident)


def cache_key_report(prefix: str, ident) -> str:
    return prefix + str(ident)


def cache_key_session(prefix: str, ident) -> str:
    return prefix + str(ident)


def cache_key_token(prefix: str, ident) -> str:
    return prefix + str(ident)
