"""Deterministic answer synthesis: the field speaks for itself.

Instead of returning one memory verbatim, `brief()` composes a
multi-fact, citation-grounded answer directly from the field's
epistemic structure — no language model anywhere:

- the lead claim is hedged according to its propagated trust
  (geometry decides the *wording*, not just the ranking)
- inferred conclusions appear with their derivation chain
- open disputes are presented as disputes, never silently resolved
- displaced history is narrated, not hidden
- every line carries its provenance

The output is what an analyst would want from a memory: not a string,
a position statement with receipts.
"""
from __future__ import annotations

from typing import Any, Dict, List

from geocell.cell import ACTIVE, CONCEPT, CONTESTED, INFERRED, SUPERSEDED, GeoCell
from geocell.field import GeoCellField


def _hedge(trust: float) -> str:
    if trust >= 0.75:
        return ""
    if trust >= 0.55:
        return "Best-supported account: "
    return "Weakly supported: "


def _cite(c: GeoCell) -> str:
    date = f", {c.date}" if c.date else ""
    return f"[#{c.id} {c.source}{date}]"


def _strip_period(text: str) -> str:
    return text.rstrip(".").rstrip()


def brief(field: GeoCellField, query: str, k: int = 12) -> Dict[str, Any]:
    """Compose a calibrated, cited mini-report answering the query."""
    hits = field.recall(query, k=k)
    if not hits:
        return {"brief": "No relevant memory.", "trust": 0.0, "lines": 0}
    field._ensure_trust()

    best = next((h for h in hits if field.cells[h["id"]].kind != CONCEPT), hits[0])
    anchor_cell = field.cells[best["id"]]
    subject = anchor_cell.subject

    related = [c for c in field.cells
               if field._subject_overlap(c.subject, subject) >= 0.75 and c.kind != CONCEPT]
    hit_rank = {h["id"]: i for i, h in enumerate(hits)}
    related.sort(key=lambda c: hit_rank.get(c.id, 999))

    lead = next((c for c in related if c.status == ACTIVE), anchor_cell)
    lines: List[str] = []
    lines.append(f"{_hedge(lead.trust)}{_strip_period(lead.content)} "
                 f"{_cite(lead)} (trust {lead.trust:.2f})")

    for c in related:
        if c.id == lead.id:
            continue
        if c.status == ACTIVE and c.kind == INFERRED:
            chain = field._derivation_chain(c)
            sources = ", ".join(sorted({s["source"] for s in chain if s["source"] != "geocell_inference"}))
            lines.append(f"By inference: {_strip_period(c.content)} "
                         f"(derived from {sources}; trust {c.trust:.2f})")
        elif c.status == ACTIVE and hit_rank.get(c.id, 999) < k:
            lines.append(f"Also established: {_strip_period(c.content)} {_cite(c)}")

    contested = [c for c in related if c.status == CONTESTED]
    if contested:
        sides = " vs ".join(f"{_strip_period(c.content)} {_cite(c)}" for c in contested[:2])
        lines.append(f"Unresolved dispute: {sides} — no side currently prevails.")

    superseded = sorted([c for c in related if c.status == SUPERSEDED],
                        key=lambda c: c.date or "")
    if superseded:
        old = superseded[0]
        lines.append(f"History: previously \"{_strip_period(old.content)}\" {_cite(old)}; "
                     f"displaced by stronger evidence.")

    open_disputes = sum(1 for c in related if c.status == CONTESTED) // 2
    closer = f"Overall trust {lead.trust:.2f}"
    closer += f"; {open_disputes} open dispute(s) in this neighborhood." if open_disputes else "; no open disputes."
    lines.append(closer)

    return {
        "brief": "\n".join(f"- {ln}" for ln in lines),
        "subject": subject,
        "trust": round(lead.trust, 4),
        "lines": len(lines),
        "method": "deterministic_synthesis(no_llm)",
    }
