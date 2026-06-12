"""Deterministic text analysis: tokens, numbers, subjects, relations.

No neural model and no training. Everything here is reproducible
across runs and machines, which keeps the geometry stable.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "to", "of", "and",
    "or", "for", "in", "on", "at", "by", "with", "as", "it", "this", "that", "from", "has",
    "have", "had", "what", "who", "when", "where", "why", "how", "current", "now", "new",
    "old", "still", "part", "into", "about", "does", "do", "did", "will", "would", "could",
    "should", "can", "may", "might", "than", "then", "there", "their", "its", "his", "her",
    "say", "says", "said", "lists", "listed",
}

NEGATORS = {"not", "no", "never", "without", "cannot", "can't", "isn't", "doesn't", "false"}

REVISION_WORDS = {"revised", "changed", "updated", "increased", "decreased", "replaced", "corrected"}

HISTORICAL_WORDS = {"old", "previous", "previously", "original", "originally", "former", "formerly", "earlier", "before", "past"}

NUMERIC_QUERY_WORDS = {"budget", "price", "amount", "rate", "revenue", "score", "much", "many"}

# Lookbehind keeps digits inside alphanumeric tokens (X1, B2B) from
# registering as numeric claims.
_NUMBER_RE = re.compile(r"(?<![a-zA-Z0-9.])\$?\d+(?:\.\d+)?\s?(?:m|b|k|million|billion|%|percent)?")

_RELATION_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("part_of", re.compile(r"^(.{2,}?)\s+(?:is|are)\s+(?:a\s+|an\s+)?part\s+of\s+(.{2,})$", re.IGNORECASE)),
    ("depends_on", re.compile(r"^(.{2,}?)\s+depends?\s+on\s+(.{2,})$", re.IGNORECASE)),
    ("requires", re.compile(r"^(.{2,}?)\s+requires?\s+(.{2,})$", re.IGNORECASE)),
]


def stable_hash(text: str) -> int:
    # FNV-1a 64-bit: deterministic across Python runs.
    h = 1469598103934665603
    for b in text.encode("utf-8", errors="ignore"):
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def raw_words(text: str) -> List[str]:
    """All words, stopwords included. Needed for cue words like 'old'
    or 'previous' that the stopword filter would otherwise eat."""
    return re.findall(r"[a-z']+", text.lower())


def tokenize(text: str) -> List[str]:
    toks = re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]*|\$?\d+(?:\.\d+)?(?:%|m|b|k| million| billion)?", text.lower())
    return [t.strip() for t in toks if t.strip() and t.strip() not in STOPWORDS]


def normalize_number(raw: str) -> Optional[float]:
    s = raw.lower().replace("$", "").strip()
    multiplier = 1.0
    if "billion" in s or s.endswith("b"):
        multiplier = 1_000_000_000.0
    elif "million" in s or s.endswith("m"):
        multiplier = 1_000_000.0
    elif s.endswith("k"):
        multiplier = 1_000.0
    elif s.endswith("%") or "percent" in s:
        multiplier = 0.01
    m = re.search(r"\d+(?:\.\d+)?", s)
    return float(m.group(0)) * multiplier if m else None


def extract_claim_values(text: str) -> List[Dict[str, Any]]:
    values = []
    for m in _NUMBER_RE.finditer(text.lower()):
        raw = m.group(0).strip()
        values.append({"raw": raw, "value": normalize_number(raw), "span": [m.start(), m.end()]})
    return values


def extract_subject(text: str) -> str:
    # Prefer a capitalized multi-word phrase (proper-noun-ish anchor).
    caps = re.findall(r"(?:[A-Z][a-zA-Z0-9-]+)(?:\s+[A-Z][a-zA-Z0-9-]+){0,3}", text)
    for c in caps:
        words = [w.lower() for w in c.split() if w.lower() not in STOPWORDS]
        if len(words) >= 2:
            return " ".join(words[:3])
    clean = [t for t in tokenize(text) if normalize_number(t) is None]
    return " ".join(clean[:2]) if len(clean) >= 2 else (clean[0] if clean else "unknown")


def normalize_entity(phrase: str) -> str:
    """Canonical key for an entity phrase: lowercase, stopword-free."""
    return " ".join(tokenize(phrase))


def _clean_phrase(phrase: str) -> str:
    phrase = phrase.strip().rstrip(".!?,;")
    phrase = re.sub(r"^(?:the|a|an)\s+", "", phrase, flags=re.IGNORECASE)
    return phrase.strip()


def extract_relations(text: str) -> List[Dict[str, str]]:
    """Extract (head, relation, tail) triples from a sentence.

    Returns dicts with both surface forms (for generating readable
    hypotheses) and normalized keys (for joining across sentences).
    """
    out = []
    sentence = text.strip().rstrip(".!?")
    for rel, pat in _RELATION_PATTERNS:
        m = pat.match(sentence)
        if not m:
            continue
        head, tail = _clean_phrase(m.group(1)), _clean_phrase(m.group(2))
        head_key, tail_key = normalize_entity(head), normalize_entity(tail)
        if head_key and tail_key and head_key != tail_key:
            out.append({
                "relation": rel,
                "head": head, "head_key": head_key,
                "tail": tail, "tail_key": tail_key,
            })
    return out
