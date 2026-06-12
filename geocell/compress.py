"""Holographic binary quantization: the serving form of the field.

A settled position is a direction on the unit hypersphere; for ranking,
only the sign pattern of its coordinates matters much. Snapping each
coordinate to its sign gives a 768-bit signature (96 bytes) whose
Hamming similarity tracks the float cosine — the same trick weight
quantization plays for neural networks, applied to memory.

float32 position: 768 x 4 bytes = 3072 B per memory
binary signature: 768 / 8     =   96 B per memory   (32x smaller)

Similarity for sign vectors is exact, not approximate:
    cos(sign(a), sign(b)) = 1 - 2 * hamming(a, b) / dims
"""
from __future__ import annotations

import base64
from typing import Tuple

import numpy as np

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)

_PROJECTIONS: dict = {}


def _projection(dims: int) -> np.ndarray:
    """Fixed random rotation (seeded, shared by every field of this
    dimensionality). Encoder output is sparse — a sentence touches a few
    dozen slots — so raw signs would agree on all the shared zeros and
    carry no signal. Rotating first spreads each vector across every
    coordinate (SimHash / LSH for cosine), making the sign pattern a
    faithful angular sketch."""
    if dims not in _PROJECTIONS:
        rng = np.random.default_rng(1469598103934665603 % (2**32))
        _PROJECTIONS[dims] = rng.standard_normal((dims, dims)).astype(np.float32)
    return _PROJECTIONS[dims]


def pack_signs(v: np.ndarray) -> np.ndarray:
    """Float vector -> packed SimHash sign bits (dims/8 bytes of uint8)."""
    v = np.asarray(v, dtype=np.float32)
    return np.packbits(_projection(v.size) @ v >= 0)


def hamming_similarity(query_bits: np.ndarray, matrix_bits: np.ndarray, dims: int) -> np.ndarray:
    """Cosine of the underlying sign vectors, computed via XOR+popcount.
    matrix_bits is (n, dims/8); returns (n,) float similarities."""
    if matrix_bits.size == 0:
        return np.zeros(0, dtype=np.float64)
    distances = _POPCOUNT[np.bitwise_xor(matrix_bits, query_bits)].sum(axis=1)
    return 1.0 - 2.0 * distances.astype(np.float64) / dims


def asymmetric_similarity(query: np.ndarray, matrix_bits: np.ndarray, dims: int) -> np.ndarray:
    """Full-precision query vs sign-quantized cells (ADC-style scoring).

    Only the stored side is quantized; the query is rotated but kept
    float, which roughly halves the sketch noise — decisive for short
    queries whose true cosines sit near the Hamming noise floor.
    Cells stay at dims/8 bytes; the +-1 expansion here is transient.
    """
    if matrix_bits.size == 0:
        return np.zeros(0, dtype=np.float64)
    q = _projection(dims) @ np.asarray(query, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if norm:
        q = q / norm
    signs = np.unpackbits(matrix_bits, axis=1)[:, :dims].astype(np.float32) * 2.0 - 1.0
    return (signs @ q) / np.sqrt(dims)


def unpack_signs(bits: np.ndarray, dims: int) -> np.ndarray:
    """Packed bits -> unit-norm float vector of +-1/sqrt(dims).

    Note: this lives in the rotated (SimHash) space, not the encoder
    space — it is a placeholder for serving-only fields, where all
    similarity runs over packed bits anyway."""
    signs = np.unpackbits(bits)[:dims].astype(np.float64) * 2.0 - 1.0
    return signs / np.sqrt(dims)


def encode_b64(bits: np.ndarray) -> str:
    return base64.b64encode(bits.tobytes()).decode("ascii")


def decode_b64(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=np.uint8).copy()


def index_footprint(n: int, dims: int) -> Tuple[int, int, float]:
    """(float_bytes, binary_bytes, compression_ratio) for n cells."""
    float_bytes = n * dims * 4
    binary_bytes = n * (dims // 8)
    return float_bytes, binary_bytes, float_bytes / max(1, binary_bytes)
