"""Configuration access with safe defaults."""


def get_timeout(settings: dict) -> int:
    return settings.get("timeout", 30)


def get_retries(settings: dict) -> int:
    return settings.get("retries", 3)


def get_pool_size(settings: dict) -> int:
    return settings.get("pool_size", 5)


def get_debug(settings: dict) -> bool:
    return settings.get("debug", False)


def get_strict_mode(settings: dict) -> bool:
    return settings.get("strict_mode", False)


def get_telemetry(settings: dict) -> bool:
    return settings.get("telemetry", False)


def get_max_conn(settings: dict) -> int:
    return settings.get("max_conn", 10)


def get_ttl(settings: dict) -> int:
    return settings.get("ttl", 60)
