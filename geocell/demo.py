"""Demo corpus and self-benchmark for the GeoCell field.

The corpus is deliberately adversarial for a similarity-only retriever:
it contains a revised figure, a stale archive copy of the old figure,
an unresolved two-source price dispute, and facts whose connection only
exists through multi-hop structure.
"""
from __future__ import annotations

import time
from typing import Any, Dict

from geocell.cell import CONTESTED, SUPERSEDED
from geocell.field import GeoCellField


def load_demo(mem: GeoCellField) -> None:
    mem.ingest("Project Orion budget is $2 million.",
               source="memo_v1", date="2026-01-01", confidence=0.72, authority=0.45)
    mem.ingest("Project Orion budget was revised to $2.7 million.",
               source="board_update", date="2026-02-01", confidence=0.95, authority=0.90)
    mem.ingest("An old Project Orion memo still lists the budget as $2 million.",
               source="archive", date="2026-01-10", confidence=0.60, authority=0.35)
    mem.ingest("Project Orion is part of the enterprise AI transformation initiative.",
               source="strategy_doc", date="2026-02-03", confidence=0.90, authority=0.80)
    mem.ingest("The enterprise AI transformation initiative depends on faster inference and lower compute cost.",
               source="technical_plan", date="2026-02-05", confidence=0.86, authority=0.75)
    mem.ingest("Project Helios budget is $900k.",
               source="finance_sheet", date="2026-01-15", confidence=0.88, authority=0.70)
    mem.ingest("Project Helios is part of the enterprise AI transformation initiative.",
               source="strategy_doc", date="2026-02-03", confidence=0.85, authority=0.80)
    # Same-day conflicting quotes from equal authorities: no winner,
    # both must surface as contested.
    mem.ingest("Vendor Atlas quoted a price of $40k.",
               source="sales_email", date="2026-03-01", confidence=0.70, authority=0.50)
    mem.ingest("Vendor Atlas quoted a price of $55k.",
               source="procurement_note", date="2026-03-01", confidence=0.70, authority=0.50)


QA_TESTS = [
    ("What is the current Project Orion budget?", "$2.7"),
    ("What did the old memo say?", "$2 million"),
    ("What initiative is Project Orion part of?", "enterprise AI"),
    ("What depends on lower compute cost?", "depends on faster inference and lower compute cost"),
    # Multi-hop: never stated anywhere in the corpus; must be inferred.
    ("Does Project Orion depend on lower compute cost?", "Project Orion depends on"),
    ("Project Helios depends on what?", "Project Helios depends on"),
]


def run_benchmark() -> Dict[str, Any]:
    """Build a fresh field, run the full pipeline, score QA accuracy and
    the epistemic state the field is expected to converge to."""
    mem = GeoCellField()
    load_demo(mem)
    hypotheses = mem.hypothesize()
    mem.propagate_trust()
    settle_info = mem.settle()
    concepts = mem.consolidate()

    qa = []
    ok = 0
    start = time.perf_counter()
    for q, expected in QA_TESTS:
        ans = mem.ask(q)
        passed = expected.lower() in ans["answer"].lower()
        ok += int(passed)
        qa.append({"query": q, "expected_contains": expected, "answer": ans["answer"],
                   "belief_status": ans.get("belief_status"), "passed": passed})
    elapsed = time.perf_counter() - start

    by_source = {c.source: c for c in mem.cells}
    structural = {
        "old_budget_superseded": by_source["memo_v1"].status == SUPERSEDED,
        "archive_copy_superseded": by_source["archive"].status == SUPERSEDED,
        "price_dispute_contested": (by_source["sales_email"].status == CONTESTED
                                    and by_source["procurement_note"].status == CONTESTED),
        "hypotheses_generated": len(hypotheses) >= 2,
        "concepts_formed": len(concepts) >= 1,
        "trust_orders_revision_over_stale": by_source["board_update"].trust > by_source["memo_v1"].trust,
        "open_contradictions_surfaced": mem.stats()["open_contradictions"] >= 1,
    }
    return {
        "qa_passed": ok,
        "qa_total": len(QA_TESTS),
        "structural_passed": sum(structural.values()),
        "structural_total": len(structural),
        "elapsed_ms": round(elapsed * 1000, 2),
        "settle_displacement": settle_info["displacement"],
        "qa": qa,
        "structural": structural,
    }
