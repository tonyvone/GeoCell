"""GeoOracle: a geometric front layer for any LLM.

The economics: an LLM call costs cents and seconds; a field query costs
microjoules and milliseconds. The oracle answers from geometry whenever
the field resonates strongly enough, and only escalates novel queries
to the LLM. Every LLM answer is ingested back, so the field absorbs
the LLM's knowledge and the hit rate climbs over time.

The epistemics: this is a cache that argues back. An LLM answer enters
the field like any memory and faces the belief lifecycle. If it
contradicts higher-trust memory, supersession fires *on ingest* and the
oracle reports the answer as quarantined — a deterministic
hallucination tripwire with an audit trail, no judge model required.

Plug in any callable `llm(prompt) -> str`.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from geocell.cell import ACTIVE, CONTESTED, RETRACTED, SUPERSEDED
from geocell.field import GeoCellField


class GeoOracle:
    def __init__(self, field: GeoCellField, llm: Optional[Callable[[str], str]] = None,
                 resonance_threshold: float = 0.55, cost_per_llm_call: float = 0.01,
                 llm_source: str = "llm_oracle"):
        self.field = field
        self.llm = llm
        self.resonance_threshold = resonance_threshold
        self.cost_per_llm_call = cost_per_llm_call
        self.llm_source = llm_source
        self.stats: Dict[str, Any] = {
            "queries": 0, "field_hits": 0, "llm_calls": 0,
            "quarantined": 0, "field_ms_total": 0.0,
        }

    def query(self, prompt: str) -> Dict[str, Any]:
        self.stats["queries"] += 1
        t0 = time.perf_counter()
        answer = self.field.ask(prompt)
        field_ms = (time.perf_counter() - t0) * 1000
        self.stats["field_ms_total"] += field_ms

        resonant = bool(answer.get("evidence")) and \
            answer["evidence"][0]["resonance"] >= self.resonance_threshold

        if resonant:
            self.stats["field_hits"] += 1
            out = {
                "answer": answer["answer"],
                "origin": "field",
                "belief_status": answer.get("belief_status"),
                "trust": answer.get("trust"),
                "latency_ms": round(field_ms, 2),
                "cost": 0.0,
            }
            if answer.get("open_contradictions"):
                out["warning"] = "answer is disputed in memory; both sides retained"
            if answer.get("reasoning_chain"):
                out["reasoning_chain"] = answer["reasoning_chain"]
            return out

        if self.llm is None:
            return {"answer": answer["answer"], "origin": "field_weak",
                    "note": "no LLM attached; best available memory returned",
                    "trust": answer.get("trust"), "cost": 0.0}

        # Escalate to the LLM, then make its answer face the field.
        text = self.llm(prompt).strip()
        self.stats["llm_calls"] += 1
        cid = self.field.ingest(text, source=self.llm_source,
                                date=time.strftime("%Y-%m-%d"),
                                confidence=0.6, authority=0.4)
        cell = self.field.cells[cid]
        out = {
            "answer": text,
            "origin": "llm",
            "ingested_as": cid,
            "cost": self.cost_per_llm_call,
        }
        if cell.status in (SUPERSEDED, RETRACTED):
            # The lifecycle rejected the LLM's claim on arrival: it
            # conflicts with stronger memory. Surface the stronger belief.
            self.stats["quarantined"] += 1
            replacement = self.field.ask(prompt)
            out.update({
                "quarantined": True,
                "reason": cell.ledger[-1].split(" ", 1)[1],
                "answer": replacement["answer"],
                "origin": "field_override",
                "llm_claim": text,
                "trust": replacement.get("trust"),
            })
        elif cell.status == CONTESTED:
            out["warning"] = "LLM answer conflicts with memory of equal strength; stored as contested"
        return out

    def report(self) -> Dict[str, Any]:
        q = max(1, self.stats["queries"])
        saved = self.stats["field_hits"] * self.cost_per_llm_call
        return {
            **self.stats,
            "field_hit_rate": round(self.stats["field_hits"] / q, 3),
            "avg_field_latency_ms": round(self.stats["field_ms_total"] / q, 2),
            "llm_cost_incurred": round(self.stats["llm_calls"] * self.cost_per_llm_call, 4),
            "llm_cost_avoided": round(saved, 4),
        }
