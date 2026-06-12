"""Deterministic hyperdimensional encoding with role binding.

A sentence becomes a point on the unit hypersphere. Coordinates are
assigned by hashing surface features (tokens, bigrams, trigrams),
role-bound features (the subject occupies different coordinates than
the same words used incidentally), magnitude buckets for numeric
claims, and a negation axis. No model weights, no token IDs.
"""
from __future__ import annotations

import math

import numpy as np

from geocell.text import (
    NEGATORS,
    extract_claim_values,
    extract_relations,
    extract_subject,
    stable_hash,
    tokenize,
)


def _slot(label: str, dims: int) -> int:
    return stable_hash(label) % dims


def _sign(label: str) -> float:
    return 1.0 if (stable_hash("sign:" + label) >> 63) == 0 else -1.0


def encode(text: str, dims: int) -> np.ndarray:
    v = np.zeros(dims, dtype=np.float32)
    toks = tokenize(text)

    # Surface features: unigrams, bigrams, trigrams.
    for i, tok in enumerate(toks):
        v[_slot("tok:" + tok, dims)] += _sign("tok:" + tok)
        if i + 1 < len(toks):
            pair = "pair:" + tok + "::" + toks[i + 1]
            v[_slot(pair, dims)] += 0.55 * _sign(pair)
        if i + 2 < len(toks):
            tri = "tri:" + tok + "::" + toks[i + 1] + "::" + toks[i + 2]
            v[_slot(tri, dims)] += 0.25 * _sign(tri)

    # Role binding: the subject of the sentence is written into its own
    # coordinate region, so "Orion" as topic differs from "Orion" in passing.
    subject = extract_subject(text)
    for tok in subject.split():
        label = "subj:" + tok
        v[_slot(label, dims)] += 0.9 * _sign(label)

    # Relation binding: structured triples occupy their own coordinates,
    # which lets a hypothesis land near the facts that produced it.
    for rel in extract_relations(text):
        label = f"rel:{rel['relation']}:{rel['head_key']}=>{rel['tail_key']}"
        v[_slot(label, dims)] += 1.1 * _sign(label)
        v[_slot("relhead:" + rel["head_key"], dims)] += 0.6
        v[_slot("reltail:" + rel["tail_key"], dims)] += 0.6

    # Numeric claims: bucketed by order of magnitude.
    for val in extract_claim_values(text):
        bucket = "value:" + str(round(math.log10(abs(val["value"] or 1) + 1), 2))
        v[_slot(bucket, dims)] += 1.25

    # Negation axis.
    if any(n in toks for n in NEGATORS):
        v[_slot("negation", dims)] -= 1.5

    norm = float(np.linalg.norm(v))
    return v / norm if norm else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0
