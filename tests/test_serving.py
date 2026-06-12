import numpy as np

from geocell import GeoCellField, SUPERSEDED
from geocell.compress import (
    asymmetric_similarity,
    hamming_similarity,
    index_footprint,
    pack_signs,
)
from geocell.demo import QA_TESTS, load_demo
from geocell.encoding import encode
from geocell.oracle import GeoOracle
from geocell.synthesis import brief


def pipeline_field() -> GeoCellField:
    mem = GeoCellField()
    load_demo(mem)
    mem.hypothesize()
    mem.propagate_trust()
    mem.settle()
    return mem


# ------------------------------------------------------------- compression

def test_sign_sketch_tracks_cosine():
    a = encode("Project Orion budget is $2 million.", 768)
    b = encode("Project Orion budget was revised to $2.7 million.", 768)
    c = encode("The weather in Lisbon is mild in October.", 768)
    ab = hamming_similarity(pack_signs(a), np.array([pack_signs(b)]), 768)[0]
    ac = hamming_similarity(pack_signs(a), np.array([pack_signs(c)]), 768)[0]
    assert ab > ac
    asym = asymmetric_similarity(a, np.vstack([pack_signs(b), pack_signs(c)]), 768)
    assert asym[0] > asym[1]


def test_quantize_is_32x_and_preserves_benchmark_qa():
    mem = pipeline_field()
    info = mem.quantize()
    assert info["compression"] == 32.0
    ok = sum(exp.lower() in mem.ask(q)["answer"].lower() for q, exp in QA_TESTS)
    assert ok == len(QA_TESTS)


def test_compact_snapshot_roundtrip_and_live_ingest(tmp_path):
    mem = pipeline_field()
    path = str(tmp_path / "field.geocell")
    info = mem.save_compact(path)
    assert info["edges_kept"] <= info["edges_total"]

    served = GeoCellField.load_compact(path)
    assert served.quantized
    assert len(served.cells) == len(mem.cells)
    ok = sum(exp.lower() in served.ask(q)["answer"].lower() for q, exp in QA_TESTS)
    assert ok == len(QA_TESTS)

    # A serving field is still alive: new evidence wires and supersedes.
    cid = served.ingest("Project Orion budget was revised to $5 million.",
                        source="late_board", date="2026-09-01",
                        confidence=0.95, authority=0.95)
    assert served.graph.degree(cid) > 0
    assert "$5 million" in served.ask("What is the current Project Orion budget?")["answer"]


def test_footprint_math():
    float_b, bin_b, ratio = index_footprint(1000, 768)
    assert float_b == 3_072_000 and bin_b == 96_000 and ratio == 32.0


# -------------------------------------------------------------- synthesis

def test_brief_composes_calibrated_cited_answer():
    mem = pipeline_field()
    out = brief(mem, "Project Orion budget")
    text = out["brief"]
    assert "$2.7 million" in text
    assert "board_update" in text                       # citations
    assert "By inference" in text                       # derived facts included
    assert "History: previously" in text                # displaced claim narrated
    assert "no open disputes" in text
    assert "Project Helios budget" not in text          # scoped to subject


def test_brief_presents_disputes_with_hedged_lead():
    mem = pipeline_field()
    out = brief(mem, "Vendor Atlas price")
    assert "Unresolved dispute" in out["brief"]
    assert "Best-supported account" in out["brief"]     # hedge from low trust
    assert out["trust"] < 0.75


# ----------------------------------------------------------------- oracle

def test_oracle_serves_known_queries_without_llm():
    mem = pipeline_field()
    calls = []
    oracle = GeoOracle(mem, llm=lambda p: calls.append(p) or "stub")
    out = oracle.query("What is the current Project Orion budget?")
    assert out["origin"] == "field" and out["cost"] == 0.0
    assert "$2.7" in out["answer"]
    assert calls == []


def test_oracle_escalates_then_absorbs():
    mem = pipeline_field()
    oracle = GeoOracle(mem, llm=lambda p: "Acme Corporation CEO is Dana Reyes.")
    first = oracle.query("Who is the CEO of Acme Corporation?")
    assert first["origin"] == "llm" and first["cost"] > 0
    second = oracle.query("Acme Corporation CEO is who?")
    assert second["origin"] == "field" and second["cost"] == 0.0
    report = oracle.report()
    assert report["llm_calls"] == 1 and report["field_hits"] >= 1
    assert report["llm_cost_avoided"] > 0


def test_oracle_quarantines_hallucination():
    mem = pipeline_field()
    oracle = GeoOracle(mem, llm=lambda p: "Project Orion budget is $9 million.")
    out = oracle.query("Orion spending plan total?")
    assert out.get("quarantined") is True
    assert out["origin"] == "field_override"
    assert "$2.7" in out["answer"]                      # stronger memory wins
    assert "$9 million" in out["llm_claim"]
    hallucination = mem.cells[-1]
    assert hallucination.status == SUPERSEDED           # audit trail retained
    assert oracle.report()["quarantined"] == 1
