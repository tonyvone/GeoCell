"""Wrappers over standard-library dependencies."""
import hashlib
import json
import math


def page_count(total: int, per_page: int) -> int:
    return math.ceil(total / per_page)


def shard_count(total: int, per_shard: int) -> int:
    return math.ceil(total / per_shard)


def batch_count(total: int, per_batch: int) -> int:
    return math.ceil(total / per_batch)


def chunk_count(total: int, per_chunk: int) -> int:
    return math.ceil(total / per_chunk)


def serialize_user(record: dict) -> str:
    return json.dumps(record, sort_keys=True)


def serialize_event(record: dict) -> str:
    return json.dumps(record, sort_keys=True)


def serialize_order(record: dict) -> str:
    return json.dumps(record, sort_keys=True)


def serialize_session(record: dict) -> str:
    return json.dumps(record, sort_keys=True)


def newest_first(items: list) -> list:
    return sorted(items, key=lambda r: r["ts"], reverse=True)


def oldest_first(items: list) -> list:
    return sorted(items, key=lambda r: r["ts"])


def by_timestamp(items: list) -> list:
    return sorted(items, key=lambda r: r["ts"])


def checksum_payload(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checksum_file(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checksum_token(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
