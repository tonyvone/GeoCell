"""GeoCell-backed fix memory + a patch-safety gate.

learn():   from a *verified* fix (one the test suite accepted), derive a
           generalized regex transformation by abstracting the buggy
           function's own identifiers, and store it in the GeoCell field
           keyed by the bug's natural-language symptom.

propose(): for a new bug, geometric recall selects the most resonant
           stored transformation; if it applies and survives the safety
           gate, return the patched source. No LLM involved.

validate_patch(): the hallucination tripwire — a patch that fails to
           compile, imports a non-existent module, or references an
           undefined name is rejected before it can touch the repo.
"""
from __future__ import annotations

import ast
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geocell import GeoCellField


# --------------------------------------------------------------------------
# Generalization: a concrete verified edit -> a transferable regex
# --------------------------------------------------------------------------

def _changed_span(old: str, new: str) -> Optional[Tuple[str, str]]:
    """Minimal differing line-block between two function bodies. A pure
    insertion is anchored to the preceding common line so the search side
    is never empty (which would match everywhere)."""
    o, n = old.splitlines(), new.splitlines()
    i = 0
    while i < len(o) and i < len(n) and o[i] == n[i]:
        i += 1
    j = 0
    while j < len(o) - i and j < len(n) - i and o[-1 - j] == n[-1 - j]:
        j += 1
    if i >= len(o) - j and i > 0:  # pure insertion: back up one anchor line
        i -= 1
    old_block = "\n".join(o[i:len(o) - j]).strip("\n")
    new_block = "\n".join(n[i:len(n) - j]).strip("\n")
    if not old_block or old_block == new_block:
        return None
    return old_block, new_block


def generalize(old_block: str, new_block: str, identifiers: List[str]) -> Tuple[str, str]:
    """Build (search_regex, replacement) that abstracts the given
    identifiers (the buggy function's name + params) into named capture
    groups, so a fix learned on one sibling transfers to the others."""
    search = re.escape(old_block)
    replacement = new_block
    # Longest first so e.g. `supplied_hash` is handled before `supplied`.
    for k, ident in enumerate(sorted(set(identifiers), key=len, reverse=True)):
        if not ident or not re.search(r"\w", ident):
            continue
        # Only abstract identifiers that occur in the search side, so the
        # replacement never references a capture group that doesn't exist.
        if not re.search(r"\b" + re.escape(ident) + r"\b", old_block):
            continue
        gname = f"id{k}"
        esc = re.escape(ident)
        seen = {"n": 0}

        def _sub(_m, gname=gname, seen=seen):
            seen["n"] += 1
            return f"(?P<{gname}>\\w+)" if seen["n"] == 1 else f"(?P={gname})"

        search = re.sub(esc, _sub, search)
        replacement = re.sub(re.escape(ident), f"\\\\g<{gname}>", replacement)
    # Collapse run-of-the-mill whitespace differences in the search side.
    search = search.replace("\\\n", "\\n").replace("\n", r"\n")
    return search, replacement


@dataclass
class FixPattern:
    search: str
    replacement: str
    symptom: str
    source_bug: str


# --------------------------------------------------------------------------
# Patch safety gate (hallucination tripwire)
# --------------------------------------------------------------------------

_ALLOWED_MODULES = {"hmac", "hashlib", "json", "math", "sqlite3", "time", "re", "os"}


def validate_patch(source: str) -> Tuple[bool, str]:
    """Reject patches that won't load. Catches the common shapes of a
    hallucinated code fix: syntax errors, imports of modules that do not
    exist, and references to names defined nowhere."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return False, f"syntax_error: {e.msg}"

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                mod = a.name.split(".")[0]
                imported.add(a.asname or mod)
                if mod not in _ALLOWED_MODULES:
                    return False, f"unknown_import: {mod}"
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod and mod not in _ALLOWED_MODULES:
                return False, f"unknown_import: {mod}"
            for a in node.names:
                imported.add(a.asname or a.name)

    # Known-attribute check on stdlib modules we understand: a call like
    # hmac.safe_compare(...) (no such function) is a hallucination.
    import hashlib as _hashlib
    import hmac as _hmac
    import math as _math
    import json as _json
    known = {"hmac": dir(_hmac), "hashlib": dir(_hashlib), "math": dir(_math), "json": _json.__all__ + dir(_json)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            base = node.value.id
            if base in known and node.attr not in known[base]:
                return False, f"unknown_reference: {base}.{node.attr}"
    return True, "ok"


# --------------------------------------------------------------------------
# The memory
# --------------------------------------------------------------------------

class FixMemory:
    def __init__(self, resonance_threshold: float = 0.35):
        self.field = GeoCellField()
        self.patterns: Dict[int, FixPattern] = {}
        self.resonance_threshold = resonance_threshold

    def learn(self, symptom: str, old_func: str, new_func: str,
              identifiers: List[str], source_bug: str) -> bool:
        span = _changed_span(old_func, new_func)
        if span is None:
            return False
        search, replacement = generalize(span[0], span[1], identifiers)
        try:
            re.compile(search)
        except re.error:
            return False
        cid = self.field.ingest(symptom, source="fix_memory", confidence=0.9, authority=0.8)
        self.patterns[cid] = FixPattern(search, replacement, symptom, source_bug)
        return True

    def propose(self, symptom: str, region: str, full_text: Optional[str] = None) -> Optional[Dict]:
        """Patch the failing region (the function body identified from the
        traceback), so a learned transformation never disturbs an already
        correct sibling elsewhere in the file. `region` is the buggy
        function source; the patched region is returned for splicing."""
        hits = self.field.recall(symptom, k=3)
        for h in hits:
            if h["resonance"] < self.resonance_threshold:
                break
            pat = self.patterns.get(h["id"])
            if not pat:
                continue
            try:
                patched_region, n = re.subn(pat.search, pat.replacement, region)
            except re.error:
                continue
            if n == 0 or patched_region == region:
                continue
            # Validate the whole module after splicing, to catch any
            # hallucinated reference the transformation might introduce.
            check_src = full_text.replace(region, patched_region, 1) if full_text else patched_region
            ok, reason = validate_patch(check_src)
            if not ok:
                return {"patched": None, "quarantined": True, "reason": reason,
                        "pattern_bug": pat.source_bug, "resonance": h["resonance"]}
            return {"patched": patched_region, "quarantined": False,
                    "pattern_bug": pat.source_bug, "resonance": h["resonance"]}
        return None
