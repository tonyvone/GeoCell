import json

import numpy as np
import pytest

from geocell import (
    ACTIVE,
    CONTESTED,
    GeoCellField,
    RETRACTED,
    SUPERSEDED,
)
from geocell.demo import load_demo, run_benchmark
from geocell.encoding import cosine, encode
from geocell.text import extract_relations, extract_subject, normalize_number


# ---------------------------------------------------------------- text layer

def test_normalize_number():
    assert normalize_number("$2.7 million") == 2_700_000.0
    assert normalize_number("900k") == 900_000.0
    assert normalize_number("3b") == 3_000_000_000.0
    assert normalize_number("15%") == pytest.approx(0.15)


def test_extract_subject_prefers_proper_nouns():
    assert extract_subject("Project Orion budget is $2 million.") == "project orion"
    assert extract_subject("Project Helios budget is $900k.") == "project helios"


def test_extract_relations():
    rels = extract_relations("Project Orion is part of the enterprise AI transformation initiative.")
    assert rels and rels[0]["relation"] == "part_of"
    assert rels[0]["head_key"] == "project orion"
    assert rels[0]["tail_key"] == "enterprise ai transformation initiative"

    rels = extract_relations("The enterprise AI transformation initiative depends on faster inference and lower compute cost.")
    assert rels and rels[0]["relation"] == "depends_on"
    assert rels[0]["head_key"] == "enterprise ai transformation initiative"


def test_encoding_is_deterministic_and_discriminative():
    a1 = encode("Project Orion budget is $2 million.", 768)
    a2 = encode("Project Orion budget is $2 million.", 768)
    b = encode("The weather in Lisbon is mild in October.", 768)
    assert np.allclose(a1, a2)
    assert cosine(a1, b) < 0.2 < cosine(a1, encode("Project Orion budget", 768))


# ------------------------------------------------------------ belief revision

def demo_field() -> GeoCellField:
    mem = GeoCellField()
    load_demo(mem)
    return mem


def test_supersession_displaces_old_budget():
    mem = demo_field()
    statuses = {c.source: c.status for c in mem.cells}
    assert statuses["memo_v1"] == SUPERSEDED
    assert statuses["archive"] == SUPERSEDED
    assert statuses["board_update"] == ACTIVE


def test_equal_evidence_stays_contested():
    mem = demo_field()
    statuses = {c.source: c.status for c in mem.cells}
    assert statuses["sales_email"] == CONTESTED
    assert statuses["procurement_note"] == CONTESTED
    assert mem.stats()["open_contradictions"] >= 1


def test_unrelated_negation_does_not_contradict():
    mem = GeoCellField()
    mem.ingest("Project Zeus is part of Group Alpha.", date="2026-01-01")
    mem.ingest("Project Zeus does not use cloud credits.", date="2026-01-02")
    assert mem.stats()["open_contradictions"] == 0


def test_ledger_records_lifecycle():
    mem = demo_field()
    memo = next(c for c in mem.cells if c.source == "memo_v1")
    assert any("superseded by" in e for e in memo.ledger)


# ----------------------------------------------------------------- inference

def test_hypothesize_creates_multihop_facts():
    mem = demo_field()
    created = mem.hypothesize()
    contents = " | ".join(c["content"] for c in created)
    assert "Project Orion depends on faster inference and lower compute cost." in contents
    assert "Project Helios depends on faster inference and lower compute cost." in contents
    # Idempotent: a second round adds nothing.
    assert mem.hypothesize() == []


def test_hypothesis_retracted_by_observed_evidence():
    mem = GeoCellField()
    mem.ingest("Project Zeus is part of Group Alpha.", date="2026-01-01", authority=0.8)
    mem.ingest("Group Alpha depends on cloud credits.", date="2026-01-02", authority=0.8)
    mem.ingest("Project Zeus does not depend on cloud credits.", date="2026-01-03", authority=0.9)
    created = mem.hypothesize()
    hyp = next(c for c in created if "Project Zeus depends on" in c["content"])
    assert mem.cells[hyp["id"]].status == RETRACTED


# --------------------------------------------------------- trust + geometry

def test_trust_orders_evidence():
    mem = demo_field()
    trust = {c.source: c.trust for c in mem.cells}
    mem.propagate_trust()
    trust = {c.source: c.trust for c in mem.cells}
    assert trust["board_update"] > trust["memo_v1"]
    assert trust["board_update"] > trust["archive"]


def test_settle_tightens_corroborated_clusters():
    mem = demo_field()
    info = mem.settle()
    assert info["displacement"] > 0
    # Mutually supporting orion cells should end up closer than
    # the unrelated vendor dispute pair is to them.
    strategy = next(c for c in mem.cells if c.source == "strategy_doc")
    tech = next(c for c in mem.cells if c.source == "technical_plan")
    atlas = next(c for c in mem.cells if c.source == "sales_email")
    p = lambda c: np.array(c.position)
    assert cosine(p(strategy), p(tech)) > cosine(p(strategy), p(atlas))


def test_recall_spreads_activation_beyond_lexical_match():
    mem = demo_field()
    hits = mem.recall("lower compute cost", k=9)
    ids_by_source = {mem.cells[h["id"]].source: i for i, h in enumerate(hits)}
    # strategy_doc never mentions compute, but is linked through the
    # initiative; it must surface above the unrelated vendor dispute.
    assert "technical_plan" in ids_by_source
    assert ids_by_source["strategy_doc"] < ids_by_source.get("sales_email", 99)


# ------------------------------------------------------------------- asking

def test_ask_resolves_revision():
    mem = demo_field()
    ans = mem.ask("What is the current Project Orion budget?")
    assert "$2.7" in ans["answer"]
    assert ans["belief_status"] == ACTIVE
    assert ans.get("superseded_history"), "should expose the displaced claims"


def test_ask_historical_query_reaches_superseded_memory():
    mem = demo_field()
    ans = mem.ask("What did the old memo say?")
    assert "$2 million" in ans["answer"]
    assert ans["belief_status"] == SUPERSEDED


def test_ask_surfaces_open_dispute():
    mem = demo_field()
    ans = mem.ask("What price did Vendor Atlas quote?")
    assert ans["belief_status"] == CONTESTED
    assert ans.get("open_contradictions")


def test_ask_inferred_answer_has_reasoning_chain():
    mem = demo_field()
    mem.hypothesize()
    ans = mem.ask("Does Project Orion depend on lower compute cost?")
    assert "Project Orion depends on" in ans["answer"]
    chain_sources = {c["source"] for c in ans["reasoning_chain"]}
    assert {"strategy_doc", "technical_plan"} <= chain_sources


# ------------------------------------------------- consolidation + plumbing

def test_consolidate_forms_concepts():
    mem = demo_field()
    concepts = mem.consolidate()
    assert concepts
    assert all(len(c["parents"]) >= 3 for c in concepts)
    assert mem.consolidate() == []  # stable on rerun


def test_timeline_shows_belief_evolution():
    mem = demo_field()
    rows = mem.timeline("Project Orion budget")
    dated = [r for r in rows if r["values"]]
    assert [r["status"] for r in dated][:2] == [SUPERSEDED, SUPERSEDED]
    assert any(r["status"] == ACTIVE and "$2.7" in r["content"] for r in dated)


def test_why_provides_full_accounting():
    mem = demo_field()
    out = mem.why(0)
    assert out["ledger"]
    assert "prior" in out["trust_breakdown"]
    assert out["neighbors"]


def test_save_load_roundtrip(tmp_path):
    mem = demo_field()
    mem.hypothesize()
    mem.propagate_trust()
    path = tmp_path / "field.json"
    mem.save(str(path))
    loaded = GeoCellField.load(str(path))
    assert len(loaded.cells) == len(mem.cells)
    assert loaded.graph.number_of_edges() == mem.graph.number_of_edges()
    assert loaded.ask("What is the current Project Orion budget?")["answer"] == \
        mem.ask("What is the current Project Orion budget?")["answer"]
    json.loads(path.read_text())  # valid JSON on disk


# ------------------------------------------------------- regression cases

def falcon_field() -> GeoCellField:
    mem = GeoCellField()
    mem.ingest("Falcon X1 launch date is March 15.",
               source="roadmap_v1", date="2026-01-05", confidence=0.8, authority=0.6)
    mem.ingest("Falcon X1 launch date was changed to April 22.",
               source="pm_announcement", date="2026-02-10", confidence=0.92, authority=0.85)
    mem.ingest("A blog rumor claims the Falcon X1 launch date is June 1.",
               source="rumor_blog", date="2026-02-15", confidence=0.3, authority=0.1)
    return mem


def test_alphanumeric_tokens_do_not_leak_claim_values():
    from geocell.text import extract_claim_values
    values = [v["value"] for v in extract_claim_values("Falcon X1 launch date is March 15.")]
    assert values == [15.0]  # the 1 in X1 must not register


def test_revision_supersedes_in_fresh_domain():
    mem = falcon_field()
    statuses = {c.source: c.status for c in mem.cells}
    assert statuses["roadmap_v1"] == SUPERSEDED
    assert statuses["pm_announcement"] == ACTIVE


def test_weak_late_claim_is_rejected_not_contesting():
    mem = falcon_field()
    rumor = next(c for c in mem.cells if c.source == "rumor_blog")
    assert rumor.status == SUPERSEDED
    assert any("superseded by #1" in e for e in rumor.ledger)
    # Conflicts involving displaced beliefs are settled, not open disputes.
    assert mem.stats()["open_contradictions"] == 0


def test_historical_query_recovers_original_claim():
    mem = falcon_field()
    ans = mem.ask("What was the original Falcon X1 launch date?")
    assert "March 15" in ans["answer"]
    assert ans["belief_status"] == SUPERSEDED


def test_different_quantities_about_same_subject_coexist():
    mem = GeoCellField()
    mem.ingest("Project Orion budget is $500k.", date="2026-01-01")
    mem.ingest("Project Orion headcount is 12.", date="2026-01-02")
    mem.ingest("Project Orion completion is 40%.", date="2026-01-03")
    assert all(c.status == ACTIVE for c in mem.cells)
    assert mem.stats()["open_contradictions"] == 0


def test_empty_field_operations_are_safe():
    mem = GeoCellField()
    assert mem.propagate_trust() == {}
    assert mem.recall("anything") == []
    assert mem.ask("anything")["confidence"] == 0.0
    assert mem.settle() == {"steps": 0, "displacement": 0.0}
    assert mem.hypothesize() == []
    assert mem.consolidate() == []
    assert mem.path("a", "b") is None


def test_benchmark_is_green():
    result = run_benchmark()
    assert result["qa_passed"] == result["qa_total"], result["qa"]
    assert result["structural_passed"] == result["structural_total"], result["structural"]
